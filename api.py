from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import EndTrialRequest, EvalPolicy, GetBenchmarkRequest, StartPlaygroundRequest, StatusRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import (
    AnswerRequest,
    ContextRequest,
    DeleteRequest,
    FindRequest,
    ListRequest,
    MkDirRequest,
    MoveRequest,
    Outcome,
    ReadRequest,
    SearchRequest,
    TreeRequest,
    WriteRequest,
)
from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, List

class TrialState(Enum):
    IDLE = "idle"
    STARTED = "started"
    ENDED = "ended"

class Trial:
    def __init__(self, client: HarnessServiceClientSync, benchmark_id: str, task_id: str):
        self.client = client
        self.benchmark_id = benchmark_id
        self.task_id = task_id
        self.id = None
        self.harness_url = None
        self.instruction = None
        self.score = None
        self.score_detail = None
        self.state = TrialState.IDLE
    
    def start(self):
        if self.state is not TrialState.IDLE:
            raise Exception(f"Trial {self.id} ({self.state}) cannot be started: trial is not in idle state")
        res = self.client.start_playground(StartPlaygroundRequest(benchmark_id=self.benchmark_id, task_id=self.task_id))
        self.id = res.trial_id
        self.harness_url = res.harness_url
        self.instruction = res.instruction
        self.state = TrialState.STARTED
    
    def end(self):
        if self.state is not TrialState.STARTED:
            raise Exception(f"Trial {self.id} ({self.state}) cannot be ended: trial is not in started state")
        res = self.client.end_trial(EndTrialRequest(trial_id=self.id))
        self.score = res.score
        self.score_detail = res.score_detail
        self.state = TrialState.ENDED

class BenchmarkState(Enum):
    IDLE = "idle"
    FETCHED = "fetched"

class Benchmark:
    def __init__(self,  client: HarnessServiceClientSync, benchmark_id: str):
        self.client = client
        self.id = benchmark_id
        self.policy = None
        self.tasks = None
        self.description = None
        self.state = BenchmarkState.IDLE

    def fetch(self):
        res = self.client.get_benchmark(GetBenchmarkRequest(benchmark_id=self.benchmark_id))
        self.policy = EvalPolicy.Name(res.policy)
        self.tasks = res.tasks
        res.description = self.description
        self.state = BenchmarkState.FETCHED


class VMCommand(BaseModel):
    pass

class RequestTreeVMCommand(VMCommand):
    level: int = Field(2, description="max tree depth, 0 means unlimited")
    root: str = Field("/", description="tree root")


class RequestFindVMCommand(VMCommand):
    name: str = Field(description="Name or pattern to search for")
    kind: Literal["all", "files", "dirs"] = Field("all", description="Type of entries to find. One of [all, files, dirs]")
    limit: int = Field(10, ge=1, le=20, description="Maximum number of results to return")


class RequestSearchVMCommand(VMCommand):
    pattern: str = Field(description="Search pattern")
    limit: int = Field(10, ge=1, le=20, description="Maximum number of results")
    root: str = Field("/", description="Root directory to search in")


class RequestListVMCommand(VMCommand):
    path: str = Field("/", description="Directory path to list")


class RequestReadVMCommand(VMCommand):
    path: str = Field(description="File path to read")
    number: bool = Field(False, description="return 1-based line numbers")
    start_line: int = Field(0, ge=0, description="1-based inclusive line number; 0 == from the first line")
    end_line: int = Field(0, ge=0, description="1-based inclusive line number; 0 == through the last line")


class RequestContextVMCommand(VMCommand):
    pass


class RequestWriteVMCommand(VMCommand):
    path: str = Field(description="File path to write")
    content: str = Field(description="Content to write")
    start_line: int = Field(0, ge=0, description="1-based inclusive line number; 0 keeps whole-file overwrite behavior")
    end_line: int = Field(0, ge=0, description="1-based inclusive line number; 0 means through the last line for ranged writes")


class RequestDeleteVMCommand(VMCommand):
    path: str = Field(description="Path to delete")


class RequestMkDirVMCommand(VMCommand):
    path: str = Field(description="Directory path to create")


class RequestMoveVMCommand(VMCommand):
    from_name: str = Field(description="Source path")
    to_name: str = Field(description="Destination path")

class OutcomeEnum(str, Enum):
    OUTCOME_UNSPECIFIED= "OUTCOME_UNSPECIFIED"
    OUTCOME_OK= "OUTCOME_OK"
    OUTCOME_DENIED_SECURITY= "OUTCOME_DENIED_SECURITY"
    OUTCOME_NONE_CLARIFICATION= "OUTCOME_NONE_CLARIFICATION"
    OUTCOME_NONE_UNSUPPORTED= "OUTCOME_NONE_UNSUPPORTED"
    OUTCOME_ERR_INTERNAL= "OUTCOME_ERR_INTERNAL"

    def to_proto(self) -> Outcome:
        return Outcome.Value(self.value)

class RequestReportTaskCompletionVMCommand(VMCommand):
    message: str = Field(description="Task completion message")
    grounding_refs: List[str] = Field(default_factory=list, description="Grounding references")
    outcome: OutcomeEnum = Field(description="Task completion outcome")


class VM:
    def __init__(self, client: PcmRuntimeClientSync):
        self.client = client

    def execute_tree_command(self, command: RequestTreeVMCommand):
        return self.client.tree(TreeRequest(root=command.root, level=command.level))

    def execute_find_command(self, command: RequestFindVMCommand):
        return self.client.find(FindRequest(root=command.root, name=command.name, type={"all": 0, "files": 1, "dirs": 2}[command.kind], limit=command.limit))

    def execute_search_command(self, command: RequestSearchVMCommand):
        return self.client.search(SearchRequest(root=command.root, pattern=command.pattern, limit=command.limit))

    def execute_list_command(self, command: RequestListVMCommand):
        return self.client.list(ListRequest(name=command.path))

    def execute_read_command(self, command: RequestReadVMCommand):
        return self.client.read(ReadRequest(path=command.path, number=command.number, start_line=command.start_line, end_line=command.end_line))

    def execute_context_command(self, command: RequestContextVMCommand):
        return self.client.context(ContextRequest())

    def execute_write_command(self, command: RequestWriteVMCommand):
        return self.client.write(WriteRequest(path=command.path, content=command.content, start_line=command.start_line, end_line=command.end_line))

    def execute_delete_command(self, command: RequestDeleteVMCommand):
        return self.client.delete(DeleteRequest(path=command.path))

    def execute_mkdir_command(self, command: RequestMkDirVMCommand):
        return self.client.mk_dir(MkDirRequest(path=command.path))

    def execute_move_command(self, command: RequestMoveVMCommand):
        return self.client.move(MoveRequest(from_name=command.from_name, to_name=command.to_name))

    def execute_report_task_completion_command(self, command: RequestReportTaskCompletionVMCommand):
        return self.client.answer(AnswerRequest(
            message=command.message,
            refs=command.grounding_refs,
            outcome=command.outcome.to_proto()
        ))

    def execute_command(self, command: VMCommand):
        match command:
            case RequestTreeVMCommand():
                return self.execute_tree_command(command)
            case RequestFindVMCommand():
                return self.execute_find_command(command)
            case RequestSearchVMCommand():
                return self.execute_search_command(command)
            case RequestListVMCommand():
                return self.execute_list_command(command)
            case RequestReadVMCommand():
                return self.execute_read_command(command)
            case RequestContextVMCommand():
                return self.execute_context_command(command)
            case RequestWriteVMCommand():
                return self.execute_write_command(command)
            case RequestDeleteVMCommand():
                return self.execute_delete_command(command)
            case RequestMkDirVMCommand():
                return self.execute_mkdir_command(command)
            case RequestMoveVMCommand():
                return self.execute_move_command(command)
            case RequestReportTaskCompletionVMCommand():
                return self.execute_report_task_completion_command(command)
            case _:
                raise ValueError(f"Unknown command type: {type(command)}")