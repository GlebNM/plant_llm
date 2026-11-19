import os
import subprocess
import sys

import asyncio


import json


import httpx2


import pytest


from pydantic_ai import Agent, NativeOutput, UsageLimits


from pydantic_ai.messages import ModelResponse, TextPart


from pydantic_ai.models.function import FunctionModel


from pydantic_ai.profiles import ModelProfile


from .graded import Card, Grade, MaterialFact, Requirement, RequirementResult, grade_payload, score, summary, validate_grade


from .runtime import BudgetExceeded, Journal, RecordedTransport, REQUEST_SCOPE, retry_transport


def run(coro):
    return asyncio.run(coro)


def payload():
    return grade_payload("Когда полить?", "Проверить влажность. Полить при подсыхании.",
        Card(ready=True, limitation="", requirements=[Requirement(id="r1", description="check", evidence_ids=["p1"])],
             acceptable_alternatives="").model_dump(), [{"id":"p1","url":"https://example.org","text":"Check soil moisture."}])


def good_grade():
    return Grade(requirements=[RequirementResult(id="r1",status="met",answer_ids=["a1"],evidence_ids=["p1"],reason="ok")],facts=[])


def test_pydantic_native_output_and_semantic_validator_correct_automatically():
    from pydantic_ai import ModelRetry, RunContext
    calls = []
    def model(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(parts=[TextPart('{"requirements":"invalid","facts":[]}')])
        grade = good_grade()
        if len(calls) == 2:
            grade.requirements[0].evidence_ids = ["invented"]
        return ModelResponse(parts=[TextPart(grade.model_dump_json())])
    agent = Agent(FunctionModel(model, profile=ModelProfile(supports_json_schema_output=True)),
                  output_type=NativeOutput(Grade), deps_type=dict, retries={"output":2})
    @agent.output_validator
    def validator(ctx: RunContext[dict], value):
        validate_grade(value, ctx.deps)
        return value
    result = agent.run_sync("judge",deps=payload(),usage_limits=UsageLimits(request_limit=3))
    assert len(calls) == 3 and score(result.output) == 2
    assert "invented" in str(calls[-1])


@pytest.mark.parametrize("failure",[429,503,"connection","timeout","protocol"])
def test_http_failures_recover_with_pydantic_transport_and_are_counted(tmp_path, failure):
    async def scenario():
        journal = Journal(tmp_path / "db.sqlite")
        journal.set_budget("test",4,10000)
        calls = 0
        def handle(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                errors={"connection":httpx2.ConnectError,"timeout":httpx2.ReadTimeout,"protocol":httpx2.RemoteProtocolError}
                if failure in errors:
                    raise errors[failure]("injected",request=request)
                return httpx2.Response(failure,headers={"Retry-After":"0"},json={"error":"temporary"})
            return httpx2.Response(200,json={"usage":{"input_tokens":20,"output_tokens":10},"answer":"ok"})
        client = httpx2.AsyncClient(transport=retry_transport(RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0),wait=lambda _:0))
        token = REQUEST_SCOPE.set(("test","job",1000))
        try:
            result=await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":100})
            assert result.json()["answer"] == "ok"
            assert calls == journal.usage("test")["requests"] == 2
            assert journal.usage("test")["charged_tokens"] == 30 if failure=="connection" else journal.usage("test")["charged_tokens"] > 30
        finally:
            REQUEST_SCOPE.reset(token)
            await client.aclose()
            journal.close()
    run(scenario())


def test_retry_after_is_honored_by_native_transport(tmp_path):
    from pydantic_ai.retries import wait_retry_after
    from types import SimpleNamespace
    response=httpx2.Response(503,headers={"retry-after":"304"},request=httpx2.Request("POST","https://example.org"))
    error=httpx2.HTTPStatusError("unavailable",request=response.request,response=response)
    state=SimpleNamespace(outcome=SimpleNamespace(exception=lambda:error))
    assert wait_retry_after(max_wait=3600)(state) == 304


def test_outage_longer_than_native_retry_burst_recovers_without_manual_restart(tmp_path):
    from .pipeline import EvaluationRun
    async def scenario():
        runner=EvaluationRun.__new__(EvaluationRun)
        runner.journal=Journal(tmp_path/"state.sqlite");runner.journal.set_budget("test",8,10000)
        runner.errors=[];runner.task_retry_delay=0;runner.status=lambda *_:None
        calls=0
        def handle(request):
            nonlocal calls
            calls+=1
            if calls<=5:raise httpx2.ConnectError("network offline",request=request)
            return httpx2.Response(200,json={"usage":{"input_tokens":20,"output_tokens":10},"ok":True})
        async with httpx2.AsyncClient(transport=retry_transport(RecordedTransport(runner.journal,httpx2.MockTransport(handle),network_pause=0),wait=lambda _:0)) as client:
            async def task(case_id):
                scope=REQUEST_SCOPE.set(("test",case_id,1000))
                try:
                    return (await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":100})).json()
                except httpx2.ConnectError as error:
                    raise RuntimeError("wrapped provider connection error") from error
                finally:REQUEST_SCOPE.reset(scope)
            result=await runner.parallel(["case"],task,"score")
            assert result[0]["ok"] and not runner.errors and calls==6
            assert runner.journal.usage("test")["charged_tokens"]==30
            assert runner.journal.usage("test")["unconfirmed_attempts"]==0
            assert await runner.parallel(["case"],task,"score")==result
            assert calls==6
        runner.journal.close()
    run(scenario())


def test_exhausted_task_recovery_count_survives_reopening(tmp_path):
    from .pipeline import EvaluationRun
    async def scenario():
        calls=0
        async def task(_):
            nonlocal calls
            calls+=1
            raise httpx2.ConnectError("offline")
        for expected in (3,4):
            runner=EvaluationRun.__new__(EvaluationRun)
            runner.journal=Journal(tmp_path/"state.sqlite")
            runner.errors=[];runner.task_retry_delay=0;runner.status=lambda *_:None
            assert await runner.parallel(["case"],task,"score")==[None]
            assert calls==expected and len(runner.errors)==1
            runner.journal.close()
    run(scenario())


def test_network_pause_is_shared_and_survives_reopening(tmp_path,monkeypatch):
    from . import runtime
    now=[1000.0];sleeps=[]
    monkeypatch.setattr(runtime.time,"time",lambda:now[0])
    async def advance(seconds):sleeps.append(seconds);now[0]+=seconds
    monkeypatch.setattr(runtime.asyncio,"sleep",advance)
    async def scenario():
        journal=Journal(tmp_path/"state.sqlite")
        transport=RecordedTransport(journal)
        transport.record_network_failure(httpx2.ConnectError("offline"))
        await transport.aclose();journal.close()
        journal=Journal(tmp_path/"state.sqlite")
        transport=RecordedTransport(journal)
        await transport.await_cooldown()
        transport.record_network_failure(httpx2.ReadTimeout("offline"))
        await transport.await_cooldown()
        assert sleeps==[30,60] and now[0]==1090
        assert journal.usage("evaluation")["requests"]==0
        await transport.aclose();journal.close()
    run(scenario())


@pytest.mark.parametrize("error_type",[httpx2.ConnectError,httpx2.ConnectTimeout,httpx2.PoolTimeout])
def test_failure_before_sending_releases_token_reservation(tmp_path,error_type):
    journal=Journal(tmp_path/"state.sqlite");journal.set_budget("test",2,10000)
    request_id=journal.reserve("test","job","f",9000,"{}")
    journal.failed(request_id,error_type("not sent"))
    assert journal.usage("test")["charged_tokens"]==0
    assert journal.usage("test")["requests"]==1
    assert journal.usage("test")["unconfirmed_attempts"]==0
    journal.close()


def test_permanent_http_errors_are_not_retried(tmp_path):
    async def scenario():
        calls=0
        def handle(request):
            nonlocal calls
            calls+=1
            return httpx2.Response(401,json={"error":"unauthorized"})
        async with httpx2.AsyncClient(transport=retry_transport(httpx2.MockTransport(handle),wait=lambda _:0)) as client:
            with pytest.raises(httpx2.HTTPStatusError):
                await client.post("https://example.org/responses")
        assert calls==1
    run(scenario())


def test_budget_is_reserved_before_network_and_survives_reopening(tmp_path):
    journal=Journal(tmp_path/"db.sqlite")
    journal.set_budget("p",1,1000)
    journal.reserve("p","j","f",500,"{}")
    journal.close()
    journal=Journal(tmp_path/"db.sqlite")
    with pytest.raises(BudgetExceeded):
        journal.reserve("p","j2","f2",1,"{}")
    assert journal.usage("p")["requests"]==1
    journal.close()


def test_token_budget_and_parallel_reservations(tmp_path):
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite")
        journal.set_budget("p",100,1000)
        async def reserve(index):
            try:
                journal.reserve("p",str(index),str(index),400,"{}")
                return True
            except BudgetExceeded:
                return False
        outcomes=await asyncio.gather(*(reserve(i) for i in range(10)))
        assert sum(outcomes)==2 and journal.usage("p")["charged_tokens"]==800
        journal.close()
    run(scenario())


def test_user_can_remove_token_ceiling_without_resetting_accounting(tmp_path):
    from .runtime import status_snapshot
    journal=Journal(tmp_path/"state.sqlite")
    journal.set_budget("evaluation",2,1000)
    journal.reserve("evaluation","first","a",500,"{}")
    with pytest.raises(BudgetExceeded):journal.reserve("evaluation","second","b",5000,"{}")
    with pytest.raises(ValueError):journal.remove_token_limit("evaluation",{})
    journal.remove_token_limit("evaluation",{"user_message":"Remove the token limit"})
    journal.close()
    journal=Journal(tmp_path/"state.sqlite")
    journal.set_budget("evaluation",2,None)
    journal.reserve("evaluation","second","b",5000,"{}")
    assert journal.usage("evaluation")["charged_tokens"]==5500
    with pytest.raises(BudgetExceeded):journal.reserve("evaluation","third","c",1,"{}")
    journal.close()
    budget=status_snapshot(tmp_path)["budgets"][0]
    assert budget["max_requests"]==2 and budget["max_tokens"] is None


def test_successful_http_response_replayed_after_process_death(tmp_path):
    db=tmp_path/"db.sqlite"
    code="""
import os,sys,json
from evaluation.runtime import Journal
j=Journal(sys.argv[1]);j.set_budget('p',2,1000)
i=j.reserve('p','j','f',300,'{}')
j.response(i,200,b'{"usage":{"input_tokens":10,"output_tokens":2}}','application/json')
os._exit(9)
"""
    env={**os.environ,"PYTHONPATH":str(__import__('pathlib').Path(__file__).resolve().parents[1])}
    child=subprocess.run([sys.executable,"-c",code,str(db)],env=env,capture_output=True)
    assert child.returncode==9,child.stderr.decode()
    journal=Journal(db)
    assert journal.replay("p","j","f")["body"] is not None
    assert journal.usage("p")["requests"]==1
    journal.close()


def test_cached_http_response_requires_no_network_or_budget(tmp_path):
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite");journal.set_budget("p",1,3000)
        calls=0
        def handle(request):
            nonlocal calls
            calls+=1
            return httpx2.Response(200,json={"usage":{"input_tokens":10,"output_tokens":2},"ok":True})
        client=httpx2.AsyncClient(transport=RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0))
        token=REQUEST_SCOPE.set(("p","j",1000))
        try:
            for _ in range(2):
                assert (await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":100})).json()["ok"]
            assert calls==1
        finally:
            REQUEST_SCOPE.reset(token);await client.aclose();journal.close()
    run(scenario())


@pytest.mark.parametrize("cached_error",[False,True])
def test_overload_inside_http_200_with_json_content_type_recovers(tmp_path,cached_error,monkeypatch):
    from .storage import digest
    async def no_cooldown(self):
        return
    monkeypatch.setattr(RecordedTransport,"await_cooldown",no_cooldown)
    async def scenario():
        journal=Journal(tmp_path/"state.sqlite");journal.set_budget("test",4,10000)
        request_body={"input":"q","stream":True,"max_output_tokens":100}
        error={"code":"server_is_overloaded","type":"service_unavailable_error",
               "message":"overloaded","headers":{"retry-after":"314"}}
        failure=("data: "+json.dumps({"type":"error","error":error})+"\n\n"+
                 "data: "+json.dumps({"type":"response.failed","response":{"error":error}})+"\n\n").encode()
        success=b'data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":2}}}\n\n'
        if cached_error:
            i=journal.reserve("test","job",digest(request_body),1000,json.dumps(request_body))
            journal.response(i,200,failure,"application/json")
        calls=0;waits=[]
        def handle(request):
            nonlocal calls
            calls+=1
            return httpx2.Response(200,content=success if cached_error or calls>1 else failure,
                                   headers={"content-type":"application/json"})
        def wait(state):
            waits.append(state.outcome.exception().response.headers["retry-after"])
            return 0
        client=httpx2.AsyncClient(transport=retry_transport(RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0),wait=wait))
        token=REQUEST_SCOPE.set(("test","job",1000))
        try:
            response=await client.post("https://example.org/responses",json=request_body)
            assert response.status_code==200 and response.headers["content-type"]=="text/event-stream"
            assert waits==["314"] and calls==(1 if cached_error else 2)
            assert journal.usage("test")["requests"]==2
            assert journal.usage("test")["failed_attempts"]==1
            await client.post("https://example.org/responses",json=request_body)
            assert calls==(1 if cached_error else 2)
        finally:
            REQUEST_SCOPE.reset(token);await client.aclose();journal.close()
    run(scenario())


def test_provider_retry_after_pauses_other_jobs_and_survives_reopen(tmp_path):
    import time
    async def scenario():
        path=tmp_path/"state.sqlite"
        journal=Journal(path);journal.set_budget("test",2,10000)
        transport=RecordedTransport(journal,httpx2.MockTransport(lambda _:httpx2.Response(200,json={"usage":{"input_tokens":1,"output_tokens":1}})))
        transport.record_cooldown(httpx2.Response(503,headers={"retry-after":"0.12"}))
        deadline=float(journal.db.execute("SELECT value FROM metadata WHERE key='provider-backoff-until'").fetchone()[0])
        journal.close();journal=Journal(path)
        starts=[]
        def handle(request):
            starts.append(time.time())
            return httpx2.Response(200,json={"usage":{"input_tokens":1,"output_tokens":1}})
        client=httpx2.AsyncClient(transport=RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0))
        token=REQUEST_SCOPE.set(("test","other-job",1000))
        try:
            await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":100})
            assert starts[0]>=deadline and journal.usage("test")["requests"]==1
        finally:
            REQUEST_SCOPE.reset(token);await client.aclose();journal.close()
    run(scenario())


def test_status_reads_live_budget_and_shared_pause_without_a_model(tmp_path):
    import time
    from .runtime import status_snapshot
    journal=Journal(tmp_path/"state.sqlite");journal.set_budget("evaluation",360,3_000_000)
    journal.reserve("evaluation","j","f",500,"{}")
    with journal.db:
        journal.db.execute("INSERT INTO metadata VALUES ('provider-backoff-until',?)",(str(time.time()+60),))
    journal.close()
    state=status_snapshot(tmp_path)
    assert state["usage"]["evaluation"]["requests"]==1
    assert state["usage"]["evaluation"]["charged_tokens"]==500
    assert state["budgets"][0]["max_requests"]==360
    assert 58<=state["provider_pause_seconds"]<=60


def test_missing_evidence_and_material_errors_have_distinct_scores():
    grade=good_grade()
    grade.facts=[MaterialFact(answer_ids=["a1"],verdict="unresolved",impact="minor",evidence_ids=[],reason="detail")]
    assert score(grade)==2
    grade.facts[0].impact="major"
    assert score(grade) is None
    grade.facts[0].verdict="contradicted";grade.facts[0].evidence_ids=["p1"]
    assert score(grade)==0


def test_partial_credit_and_unknown_denominator():
    grade=good_grade()
    grade.requirements[0].status="partial"
    assert score(grade)==1
    rows=[{"score":2,"grade":good_grade().model_dump()},{"score":None,"grade":grade.model_dump()}]
    assert summary(rows,2)["score_bounds_100"]==[50,100]
    assert summary(rows,3)["score_bounds_100"] is None


def test_identical_unknown_answers_have_zero_paired_change():
    from .metrics import compare_scores
    row={"id":"a","question":"q","answer":"identical answer","card_version":"c",
         "judge_signature":"j","source_group_id":"g","score":None}
    report=compare_scores([row],[dict(row)],samples=100)
    assert report["delta_score_bounds"]==[0,0] and report["paired_bootstrap_with_unresolved_95_bounds"]==[0,0]
    assert report["unchanged_answers"]==1
    changed=compare_scores([row],[{**row,"answer":"changed answer"}],samples=100)
    assert changed["delta_score_bounds"]==[-100,100] and changed["unchanged_answers"]==0


def test_unknown_ids_and_missing_requirements_rejected():
    from pydantic_ai import ModelRetry
    grade=good_grade();grade.requirements[0].answer_ids=["made-up"]
    with pytest.raises(ModelRetry):validate_grade(grade,payload())
    with pytest.raises(ModelRetry):validate_grade(Grade(requirements=[],facts=[]),payload())


def test_codex_budget_reserves_documented_output_ceiling(tmp_path):
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite");journal.set_budget("p",10,120000)
        calls=[]
        def handle(request):
            calls.append(request)
            return httpx2.Response(200,json={})
        async with httpx2.AsyncClient(transport=RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0)) as client:
            token=REQUEST_SCOPE.set(("p","j",1000))
            try:
                with pytest.raises(BudgetExceeded):
                    await client.post("https://example.org/responses",json={"model":"gpt-6.1-sol","input":"q"})
                assert calls==[]
            finally:
                REQUEST_SCOPE.reset(token)
        journal.close()
    run(scenario())


def test_truncated_sse_is_retried_and_never_cached_as_success(tmp_path):
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite");journal.set_budget("p",4,10000)
        calls=0
        def handle(request):
            nonlocal calls
            calls+=1
            body=b'data: {"type":"response.created"}\n\n'
            if calls>1:
                body+=b'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":2}}}\n\n'
            return httpx2.Response(200,content=body,headers={"content-type":"text/event-stream"})
        async with httpx2.AsyncClient(transport=retry_transport(RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0),wait=lambda _:0)) as client:
            token=REQUEST_SCOPE.set(("p","j",1000))
            try:
                response=await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":100})
                assert "response.completed" in response.text and calls==2
                await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":100})
                assert calls==2
            finally:REQUEST_SCOPE.reset(token)
        journal.close()
    run(scenario())


def test_comparison_keeps_uncertainty_and_rejects_changed_benchmark():
    from .metrics import compare_scores
    row={"id":"a","question":"q","card_version":"c","judge_signature":"j","source_group_id":"g","score":2}
    result=compare_scores([row],[{**row,"score":None}],samples=100)
    assert result["delta_score_bounds"]==[-100,0]
    assert result["paired_bootstrap_with_unresolved_95_bounds"]==[-100,0]
    with pytest.raises(ValueError):compare_scores([row],[{**row,"card_version":"changed"}],samples=100)
    with pytest.raises(ValueError):compare_scores([row],[{**row,"judge_signature":"changed"}],samples=100)


def test_parallel_requests_wait_for_reserved_budget_to_be_released(tmp_path):
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite");journal.set_budget("p",2,2300)
        active=peak=0
        async def handle(request):
            nonlocal active,peak
            active+=1;peak=max(peak,active)
            await asyncio.sleep(.02)
            active-=1
            return httpx2.Response(200,json={"usage":{"input_tokens":10,"output_tokens":2}})
        async with httpx2.AsyncClient(transport=RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0)) as client:
            async def request(index):
                token=REQUEST_SCOPE.set(("p",str(index),1000))
                try:
                    return await client.post("https://example.org/responses",json={"input":"q","max_output_tokens":500})
                finally:REQUEST_SCOPE.reset(token)
            results=await asyncio.gather(request(0),request(1))
        assert len(results)==2 and peak==1
        assert journal.usage("p")["requests"]==2 and journal.usage("p")["charged_tokens"]==24
        journal.close()
    run(scenario())


def test_300_answers_correct_validation_errors_and_resume_without_calls(tmp_path):
    from pydantic_ai.messages import UserPromptPart
    from .graded import Grader
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite")
        attempts={}
        def model(messages,info):
            first=next(part.content for message in messages for part in message.parts if isinstance(part,UserPromptPart))
            question=json.loads(first)["question"]
            attempts[question]=attempts.get(question,0)+1
            grade=good_grade()
            if int(question[1:])%17==0 and attempts[question]==1:
                grade.requirements[0].evidence_ids=["invented"]
            return ModelResponse(parts=[TextPart(grade.model_dump_json())])
        backend=FunctionModel(model,profile=ModelProfile(supports_json_schema_output=True))
        grader=Grader(journal,backend=backend)
        semaphore=asyncio.Semaphore(6)
        async def evaluate(index):
            async with semaphore:
                data=payload();data["question"]=f"q{index}"
                return await grader.call("test","judge",data)
        first=await asyncio.gather(*(evaluate(i) for i in range(300)))
        calls=sum(attempts.values())
        second=await asyncio.gather(*(evaluate(i) for i in range(300)))
        assert calls==318 and sum(attempts.values())==calls
        assert all(score(value)==2 and not cached for value,cached in first)
        assert all(score(value)==2 and cached for value,cached in second)
        assert journal.db.execute("SELECT count(*) FROM jobs WHERE state='done'").fetchone()[0]==300
        await grader.close();journal.close()
        candidate=Journal(tmp_path/"candidate.sqlite")
        candidate.import_results(tmp_path/"db.sqlite",kind="judge")
        grader=Grader(candidate,backend=backend)
        data=payload();data["question"]="q0"
        _,cached=await grader.call("evaluation","judge",data)
        assert cached and sum(attempts.values())==318
        data["answer"][0]["text"]="Changed answer"
        _,cached=await grader.call("evaluation","judge",data)
        assert not cached and sum(attempts.values())==319
        await grader.close();candidate.close()
    run(scenario())




def test_major_unverified_claim_requires_concrete_decision_impact():
    from pydantic_ai import ModelRetry
    grade=good_grade()
    grade.facts=[MaterialFact(answer_ids=["a1"],verdict="unresolved",impact="major",evidence_ids=[],reason="no source")]
    with pytest.raises(ModelRetry,match="decision_impact"):
        validate_grade(grade,payload())
    grade.facts[0].decision_impact="The user would water before checking moisture."
    validate_grade(grade,payload())


@pytest.mark.parametrize("defect",["missing","answer_quote","source_quote","evidence_id"])
def test_contradiction_proof_is_corrected_by_pydantic_validator(defect):
    from .graded import ContradictionProof
    from pydantic_ai import RunContext
    data=payload()
    data["answer"]=[{"id":"a1","text":"Полейте сразу, не проверяя почву."}]
    data["passages"][0]["text"]="Check soil moisture before watering."
    proof=ContradictionProof(answer_quote=data["answer"][0]["text"],evidence_id="p1",
        source_quote=data["passages"][0]["text"],applicability="Same watering decision",
        competing_interpretation="Moisture could have been checked earlier",
        why_incompatible="The answer explicitly says not to check")
    value=good_grade()
    value.requirements[0].status="missing"
    value.facts=[MaterialFact(answer_ids=["a1"],verdict="contradicted",impact="major",evidence_ids=["p1"],
        reason="Skips required moisture check",decision_impact="Watering potentially wet soil",contradiction=proof)]
    calls=[]
    def model(messages,info):
        calls.append(messages)
        result=value.model_copy(deep=True)
        if len(calls)==1:
            if defect=="missing":result.facts[0].contradiction=None
            else:setattr(result.facts[0].contradiction,defect,"invented")
        return ModelResponse(parts=[TextPart(result.model_dump_json())])
    agent=Agent(FunctionModel(model,profile=ModelProfile(supports_json_schema_output=True)),
        output_type=NativeOutput(Grade),deps_type=dict,retries={"output":2})
    @agent.output_validator
    def validator(ctx:RunContext[dict],output):
        validate_grade(output,ctx.deps)
        return output
    result=agent.run_sync("judge",deps=data,usage_limits=UsageLimits(request_limit=3))
    assert len(calls)==2 and result.output==value and score(result.output)==0


@pytest.mark.parametrize("large_invalid_output",[False,True])
def test_codex_sse_native_correction_transport_retry_and_crash_replay_together(tmp_path,large_invalid_output):
    from openai import AsyncOpenAI
    from pydantic_ai.models import create_async_httpx2_client
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.openai_codex import OpenAICodexProvider
    from .graded import Grader
    async def scenario():
        journal=Journal(tmp_path/"db.sqlite");journal.set_budget("test",6,1_000_000)
        calls=0
        def handle(request):
            nonlocal calls
            calls+=1
            if calls==1:
                return httpx2.Response(503,json={"error":"injected outage"})
            grade=good_grade()
            if calls==2:
                grade.requirements[0].evidence_ids=["invented"]
                if large_invalid_output:
                    grade.requirements[0].reason="detail "*13000
            if calls==3 and large_invalid_output:
                from .runtime import token_count
                assert 14000<int(token_count(json.loads(request.content))*1.2)+256<30000
            output=grade.model_dump_json()
            part={"type":"output_text","text":output,"annotations":[]}
            item={"id":"msg_local","type":"message","role":"assistant","status":"completed","content":[part]}
            response={"id":f"resp_local_{calls}","object":"response","created_at":1,"model":"gpt-6.1-sol",
                      "status":"in_progress","output":[]}
            events=[{"type":"response.created","response":response},
                    {"type":"response.output_item.added","output_index":0,"item":{**item,"status":"in_progress","content":[]}},
                    {"type":"response.content_part.added","output_index":0,"item_id":item["id"],"content_index":0,"part":{**part,"text":""}},
                    {"type":"response.output_text.delta","output_index":0,"item_id":item["id"],"content_index":0,"delta":output},
                    {"type":"response.output_text.done","output_index":0,"item_id":item["id"],"content_index":0,"text":output},
                    {"type":"response.output_item.done","output_index":0,"item":item},
                    {"type":"response.completed","response":{**response,"status":"completed","output":[item],
                        "usage":{"input_tokens":30,"output_tokens":20,"total_tokens":50}}}]
            body="".join("data: "+json.dumps({**event,"sequence_number":i})+"\n\n" for i,event in enumerate(events))
            return httpx2.Response(200,text=body,headers={"content-type":"text/event-stream"})
        http=create_async_httpx2_client(transport=retry_transport(RecordedTransport(journal,httpx2.MockTransport(handle),network_pause=0),wait=lambda _:0))
        client=AsyncOpenAI(api_key="local-test-only",base_url="https://example.org",http_client=http,max_retries=0)
        backend=OpenAICodexModel("gpt-6.1-sol",provider=OpenAICodexProvider(openai_client=client))
        grader=Grader(journal,backend=backend)
        first,cached=await grader.call("test","judge",payload())
        assert not cached and score(first)==2 and calls==3
        with journal.db:
            journal.db.execute("UPDATE jobs SET state='running',result=NULL")
        recovered,cached=await grader.call("test","judge",payload())
        assert not cached and score(recovered)==2 and calls==3
        assert journal.usage("test")["requests"]==3
        await grader.close();journal.close()
    run(scenario())
