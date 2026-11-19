import contextvars
import asyncio
import json
import sqlite3
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx2
import tiktoken
from pydantic_ai.models import create_async_httpx2_client
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.providers.openai_codex import OpenAICodexProvider
from pydantic_ai.retries import AsyncHTTPX2TenacityTransport, RetryConfig, wait_retry_after
from tenacity import retry_if_exception, stop_after_attempt, wait_random_exponential

from .storage import digest

REQUEST_SCOPE = contextvars.ContextVar("evaluation_request_scope", default=None)
UNLIMITED_TOKENS = 2**63-1


class BudgetExceeded(RuntimeError):
    pass


def token_count(value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))


def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def status_snapshot(directory):
    path=directory/"status.json"
    report=json.loads(path.read_text()) if path.exists() else {"stage":"not_started","errors":[]}
    database=directory/"state.sqlite"
    if not database.exists():
        return report
    connection=sqlite3.connect(f"file:{database.resolve()}?mode=ro",uri=True)
    connection.row_factory=sqlite3.Row
    try:
        report["usage"]={row["phase"]:dict(row) for row in connection.execute("""
            SELECT phase,count(*) requests,coalesce(sum(charged),0) charged_tokens,
            coalesce(sum(input_tokens),0) reported_input_tokens,coalesce(sum(output_tokens),0) reported_output_tokens,
            sum(status>=400 OR error IS NOT NULL) failed_attempts,
            sum(status IS NULL AND coalesce(error,'') NOT IN ('ConnectError','ConnectTimeout','PoolTimeout')) unconfirmed_attempts
            FROM wire GROUP BY phase""")}
        report["budgets"]=[dict(row) for row in connection.execute("SELECT * FROM budgets")]
        for budget in report["budgets"]:
            if budget["max_tokens"]==UNLIMITED_TOKENS:
                budget["max_tokens"]=None
        report["jobs"]=[dict(row) for row in connection.execute("SELECT phase,kind,state,count(*) count FROM jobs GROUP BY phase,kind,state")]
        row=connection.execute("SELECT value FROM metadata WHERE key='provider-backoff-until'").fetchone()
        report["provider_pause_seconds"]=max(0,round(float(row[0])-time.time())) if row else 0
    finally:
        connection.close()
    return report


class Journal:
    def __init__(self, path: Path):
        self.active = {}
        self.changed = asyncio.Event()
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS budgets (
                phase TEXT PRIMARY KEY, max_requests INTEGER NOT NULL, max_tokens INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS wire (
                id INTEGER PRIMARY KEY, phase TEXT, job TEXT, fingerprint TEXT, started REAL,
                reserved INTEGER, charged INTEGER, input_tokens INTEGER, output_tokens INTEGER,
                status INTEGER, body BLOB, content_type TEXT, error TEXT, request TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                key TEXT PRIMARY KEY, phase TEXT, kind TEXT, input TEXT, state TEXT,
                result TEXT, error TEXT, updated REAL
            );
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
        """)

    def close(self):
        self.db.close()

    def set_budget(self, phase, requests, tokens):
        tokens=UNLIMITED_TOKENS if tokens is None else tokens
        existing = self.db.execute("SELECT * FROM budgets WHERE phase=?", (phase,)).fetchone()
        if existing and (existing["max_requests"], existing["max_tokens"]) != (requests, tokens):
            raise ValueError(f"Budget for {phase} is frozen; use a new run for a changed budget")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO budgets VALUES (?,?,?)", (phase, requests, tokens))

    def remove_token_limit(self,phase,authorization):
        if not authorization or not str(authorization.get("user_message","")).strip():
            raise ValueError("Removing a token limit requires recorded user authorization")
        with self.db:
            row=self.db.execute("SELECT max_tokens FROM budgets WHERE phase=?",(phase,)).fetchone()
            if row is None:
                raise ValueError("Unknown budget phase")
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",
                (f"budget-authorization:{phase}",json.dumps({"previous_max_tokens":row[0],"authorization":authorization},ensure_ascii=False)))
            self.db.execute("UPDATE budgets SET max_tokens=? WHERE phase=?",(UNLIMITED_TOKENS,phase))

    def usage(self, phase):
        row = self.db.execute("""SELECT count(*) requests, coalesce(sum(charged),0) charged_tokens,
            coalesce(sum(input_tokens),0) reported_input_tokens,
            coalesce(sum(output_tokens),0) reported_output_tokens,
            sum(status>=400 OR error IS NOT NULL) failed_attempts,
            sum(status IS NULL AND coalesce(error,'') NOT IN ('ConnectError','ConnectTimeout','PoolTimeout')) unconfirmed_attempts
            FROM wire WHERE phase=?""", (phase,)).fetchone()
        return dict(row)

    def reserve(self, phase, job, fingerprint, reservation, request):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            budget = self.db.execute("SELECT * FROM budgets WHERE phase=?", (phase,)).fetchone()
            usage = self.usage(phase)
            if not budget or usage["requests"] >= budget["max_requests"] or (budget["max_tokens"]!=UNLIMITED_TOKENS and usage["charged_tokens"] + reservation > budget["max_tokens"]):
                raise BudgetExceeded(f"Budget exhausted for {phase}: {usage}")
            cursor = self.db.execute("""INSERT INTO wire
                (phase,job,fingerprint,started,reserved,charged,request) VALUES (?,?,?,?,?,?,?)""",
                (phase, job, fingerprint, time.time(), reservation, reservation, request))
            self.db.commit()
            return cursor.lastrowid
        except BaseException:
            self.db.rollback()
            raise

    def response(self, request_id, status, body, content_type):
        usage = None
        try:
            data = json.loads(body)
            usage = data.get("usage")
        except (ValueError, AttributeError):
            for line in body.splitlines():
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:])
                        usage = data.get("response", {}).get("usage") or usage
                    except (ValueError, AttributeError):
                        pass
        input_tokens = usage.get("input_tokens", 0) if usage else None
        output_tokens = usage.get("output_tokens", 0) if usage else None
        charged = (input_tokens + output_tokens) if usage else None
        with self.db:
            self.db.execute("""UPDATE wire SET status=?,body=?,content_type=?,input_tokens=?,output_tokens=?,
                charged=coalesce(?,charged) WHERE id=?""",
                (status, body, content_type, input_tokens, output_tokens, charged, request_id))

    def failed(self, request_id, error):
        with self.db:
            self.db.execute("UPDATE wire SET error=? WHERE id=?", (type(error).__name__, request_id))
            if isinstance(error,(httpx2.ConnectError,httpx2.ConnectTimeout,httpx2.PoolTimeout)):
                self.db.execute("UPDATE wire SET charged=0,input_tokens=0,output_tokens=0 WHERE id=? AND status IS NULL",(request_id,))

    def replay(self, phase, job, fingerprint):
        return self.db.execute("""SELECT * FROM wire WHERE phase=? AND job=? AND fingerprint=?
            AND status BETWEEN 200 AND 299 AND body IS NOT NULL AND error IS NULL ORDER BY id DESC LIMIT 1""",
            (phase, job, fingerprint)).fetchone()

    def start(self, key, phase, kind, payload):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                            (key, phase, kind, json.dumps(payload, ensure_ascii=False), "pending", None, None, time.time()))
            self.db.execute("UPDATE jobs SET state='running',error=NULL,updated=? WHERE key=? AND state!='done'",
                            (time.time(), key))

    def result(self, key):
        row = self.db.execute("SELECT result FROM jobs WHERE key=? AND state='done'", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def import_results(self, path, kind):
        source=sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro",uri=True)
        try:
            rows=source.execute("SELECT * FROM jobs WHERE state='done' AND kind=?",(kind,)).fetchall()
            with self.db:
                self.db.executemany("INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?,?,?)",rows)
        finally:
            source.close()

    def finish(self, key, result):
        with self.db:
            self.db.execute("UPDATE jobs SET state='done',result=?,error=NULL,updated=? WHERE key=?",
                            (json.dumps(result, ensure_ascii=False), time.time(), key))

    def fail(self, key, error):
        with self.db:
            self.db.execute("UPDATE jobs SET state='error',error=?,updated=? WHERE key=?",
                            (f"{type(error).__name__}: {str(error)[:600]}", time.time(), key))


class RecordedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, journal: Journal, wrapped=None, network_pause=30):
        self.journal = journal
        self.wrapped = wrapped or httpx2.AsyncHTTPTransport()
        self.network_pause=network_pause

    def record_network_failure(self,error):
        if not isinstance(error,(httpx2.NetworkError,httpx2.TimeoutException,httpx2.RemoteProtocolError)):
            return
        row=self.journal.db.execute("SELECT value FROM metadata WHERE key='network-failures'").fetchone()
        count=(int(row[0]) if row else 0)+1
        row=self.journal.db.execute("SELECT value FROM metadata WHERE key='provider-backoff-until'").fetchone()
        deadline=max(float(row[0]) if row else 0,time.time()+min(300,self.network_pause*2**min(count-1,5)))
        with self.journal.db:
            self.journal.db.execute("INSERT OR REPLACE INTO metadata VALUES ('network-failures',?)",(str(count),))
            self.journal.db.execute("INSERT OR REPLACE INTO metadata VALUES ('provider-backoff-until',?)",(str(deadline),))

    def record_cooldown(self,response):
        if response.status_code not in {429,503}:
            return
        value=response.headers.get("retry-after")
        try:
            delay=float(value) if value is not None else float(response.headers.get("retry-after-ms","0"))/1000
        except ValueError:
            try:
                delay=parsedate_to_datetime(value).timestamp()-time.time()
            except (TypeError,ValueError):
                return
        row=self.journal.db.execute("SELECT value FROM metadata WHERE key='provider-backoff-until'").fetchone()
        deadline=max(float(row[0]) if row else 0,time.time()+max(0,min(delay,3600)))
        with self.journal.db:
            self.journal.db.execute("INSERT OR REPLACE INTO metadata VALUES ('provider-backoff-until',?)",(str(deadline),))

    async def await_cooldown(self):
        while True:
            row=self.journal.db.execute("SELECT value FROM metadata WHERE key='provider-backoff-until'").fetchone()
            delay=float(row[0])-time.time() if row else 0
            if delay<=0:
                return
            await asyncio.sleep(delay)

    @staticmethod
    def checked_response(request,status,body,headers,stream):
        if status>=400 or not stream:
            return httpx2.Response(status,content=body,headers=headers,request=request)
        events=[]
        for line in body.splitlines():
            if line.startswith(b"data: "):
                try:
                    events.append(json.loads(line[6:]))
                except ValueError:
                    pass
        errors=[e.get("error") or e.get("response",{}).get("error") for e in events
                if e.get("type") in {"error","response.failed"}]
        error=next((e for e in errors if e),None)
        if error:
            code=error.get("code") or error.get("type")
            status=503 if code in {"server_is_overloaded","service_unavailable_error"} else 429 if code in {"rate_limit_exceeded","rate_limit_error"} else 400 if code in {"invalid_request_error","context_length_exceeded"} else 502
            extra=error.get("headers",{})
            retry_headers={k:str(v) for k,v in extra.items() if k.lower() in {"retry-after","retry-after-ms"}}
            return httpx2.Response(status,json={"error":error},headers=retry_headers,request=request)
        if not any(e.get("type")=="response.completed" for e in events):
            raise httpx2.RemoteProtocolError("Stream ended without a completed response",request=request)
        return httpx2.Response(status,content=body,headers={**dict(headers),"content-type":"text/event-stream"},request=request)

    async def handle_async_request(self, request):
        scope = REQUEST_SCOPE.get()
        if not scope or not request.url.path.endswith("/responses"):
            return await self.wrapped.handle_async_request(request)
        phase, job, input_limit = scope
        raw = await request.aread()
        payload = json.loads(raw)
        canonical = {k: v for k, v in payload.items() if k not in {"prompt_cache_key", "metadata"}}
        fingerprint = digest(canonical)
        cached = self.journal.replay(phase, job, fingerprint)
        if cached:
            try:
                response=self.checked_response(request,cached["status"],cached["body"],
                    {"content-type":cached["content_type"]},payload.get("stream") or "event-stream" in cached["content_type"])
                if response.status_code>=400:
                    self.journal.failed(cached["id"],RuntimeError("Provider error inside cached stream"))
                    self.record_cooldown(response)
                return response
            except BaseException as error:
                self.journal.failed(cached["id"],error)
                raise
        estimated_input = int(token_count(canonical) * 1.2) + 256
        if estimated_input > input_limit:
            raise BudgetExceeded(f"Request input estimate {estimated_input} exceeds {input_limit}")
        output_limit = payload.get("max_output_tokens")
        if not output_limit:
            if payload.get("model") != "gpt-6.1-sol":
                raise BudgetExceeded("Model output ceiling is not registered")
            output_limit = 128000
        input_reservation = max(estimated_input, len(json.dumps(canonical,ensure_ascii=False).encode()) + 1024)
        while True:
            await self.await_cooldown()
            self.journal.changed.clear()
            try:
                request_id = self.journal.reserve(phase, job, fingerprint, input_reservation + output_limit,
                                                  json.dumps(canonical, ensure_ascii=False))
                break
            except BudgetExceeded:
                budget=self.journal.db.execute("SELECT max_requests FROM budgets WHERE phase=?",(phase,)).fetchone()
                if not budget or self.journal.usage(phase)["requests"]>=budget[0] or phase not in self.journal.active.values():
                    raise
                await self.journal.changed.wait()
        self.journal.active[request_id]=phase
        try:
            response = await self.wrapped.handle_async_request(request)
            try:
                body = await response.aread()
            finally:
                await response.aclose()
            self.journal.response(request_id, response.status_code, body, response.headers.get("content-type", "application/json"))
            with self.journal.db:
                self.journal.db.execute("DELETE FROM metadata WHERE key='network-failures'")
            checked=self.checked_response(request,response.status_code,body,response.headers,
                payload.get("stream") or "event-stream" in response.headers.get("content-type", ""))
            if checked.status_code>=400 and response.status_code<400:
                self.journal.failed(request_id,RuntimeError("Provider error inside stream"))
            self.record_cooldown(checked)
            return checked
        except BaseException as error:
            self.journal.failed(request_id, error)
            self.record_network_failure(error)
            raise
        finally:
            self.journal.active.pop(request_id,None)
            self.journal.changed.set()

    async def aclose(self):
        await self.wrapped.aclose()


def retryable(error):
    return isinstance(error, (httpx2.TimeoutException, httpx2.NetworkError, httpx2.RemoteProtocolError)) or (
        isinstance(error, httpx2.HTTPStatusError) and error.response.status_code in {408, 409, 429, 500, 502, 503, 504})


def transient_failure(error):
    seen=set()
    while error is not None and id(error) not in seen:
        if isinstance(error,BudgetExceeded):
            return False
        seen.add(id(error))
        if retryable(error) or getattr(error,"status_code",None) in {408,409,429,500,502,503,504}:
            return True
        error=error.__cause__ or error.__context__
    return False


def retry_transport(wrapped=None, wait=None):
    return AsyncHTTPX2TenacityTransport(
        RetryConfig(retry=retry_if_exception(retryable), stop=stop_after_attempt(4), reraise=True,
                    wait=wait if wait is not None else wait_retry_after(
                        fallback_strategy=wait_random_exponential(multiplier=2, max=60), max_wait=3600)),
        wrapped=wrapped, validate_response=lambda response: response.raise_for_status() if response.status_code>=400 else None,
    )


def make_model(name, journal=None):
    transport = retry_transport(RecordedTransport(journal) if journal else None)
    client = create_async_httpx2_client(transport=transport)
    return OpenAICodexModel(name, provider=OpenAICodexProvider(http_client=client))
