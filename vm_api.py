from __future__ import annotations

import json
import shlex
from enum import Enum
from dataclasses import dataclass
from typing import Iterable, Protocol, TypeVar, Generic, Literal, List

from pydantic import BaseModel, Field

from google.protobuf.message import Message
from google.protobuf.json_format import MessageToDict

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import EndTrialRequest, EvalPolicy, GetBenchmarkRequest, StartPlaygroundRequest, StatusRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import (
    AnswerRequest,
    ContextRequest,
    DeleteRequest,
    FindRequest,
    ListRequest,
    ListResponse,
    MkDirRequest,
    MoveRequest,
    Outcome,
    ReadRequest,
    SearchRequest,
    TreeRequest,
    WriteRequest,
    ReadResponse
)


class TreeNode(Protocol):
    name: str
    children: Iterable[TreeNode]


T = TypeVar("T", bound=TreeNode)

class TreeFormatter(Generic[T]):
    def __init__(self, root: T) -> None:
        self._root = root

    def format(self) -> str:
        return "\n".join(self._lines())

    def __str__(self) -> str:
        return self.format()

    def _lines(self) -> Iterable[str]:
        if not self._root.name:
            yield "."
            return
        yield self._root.name
        yield from self._render_children(self._root, prefix="")

    def _render_children(self, node: TreeNode, prefix: str) -> Iterable[str]:
        children = list(node.children)
        for idx, child in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└── " if is_last else "├── "
            yield f"{prefix}{connector}{child.name}"
            child_prefix = prefix + ("    " if is_last else "│   ")
            yield from self._render_children(child, child_prefix)


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


@dataclass
class Benchmark:
    id: str
    policy: str
    description: str
    tasks: list

def fetch_benchmark(client: HarnessServiceClientSync, benchmark_id: str) -> Benchmark:
    res = client.get_benchmark(GetBenchmarkRequest(benchmark_id=benchmark_id))
    return Benchmark(
        id=benchmark_id,
        policy=EvalPolicy.Name(res.policy),
        description=res.description,
        tasks=list(res.tasks),
    )


class VMCommand(BaseModel):
    pass

class RequestTreeVMCommand(VMCommand):
    level: int = Field(2, description="max tree depth, 0 means unlimited")
    root: str = Field("/", description="tree root")


class RequestFindVMCommand(VMCommand):
    name: str = Field(description="Name or pattern to search for")
    kind: Literal["all", "files", "dirs"] = Field("all", description="Type of entries to find. One of [all, files, dirs]")
    limit: int = Field(10, ge=1, le=20, description="Maximum number of results to return")
    root: str = Field("/", description="search root")


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
            

class VMResponseFormatter:
    @staticmethod
    def format(command: VMCommand, response: Message) -> str:
        if response is None:
            return "{}"
        match command:
            case RequestListVMCommand():
                return VMResponseFormatter.format_list_response(command, response)
            case RequestFindVMCommand():
                return VMResponseFormatter.format_find_response(command, response)
            case RequestTreeVMCommand():
                return VMResponseFormatter.format_tree_response(command, response)
            case RequestReadVMCommand():
                return VMResponseFormatter.format_read_response(command, response)
            case RequestSearchVMCommand():
                return VMResponseFormatter.format_search_response(command, response)
            case _:
                return json.dumps(MessageToDict(response), indent=2)

    @staticmethod
    def format_list_response(command: RequestListVMCommand, response: ListResponse) -> str:
        body = "\n".join(
            f"{entry.name}/" if entry.is_dir else entry.name for entry in response.entries
        ) or "."
        return VMResponseFormatter._format_command(f"ls {command.path}", body)
    
    @staticmethod
    def format_tree_response(command: RequestTreeVMCommand, response: ListResponse) -> str:
        body = TreeFormatter(response.root).format()
        level_arg = f" -L {command.level}" if command.level > 0 else ""
        return VMResponseFormatter._format_command(f"tree{level_arg} {command.root}", body)
    
    @staticmethod
    def format_read_response(command: RequestReadVMCommand, response: ReadResponse) -> str:
        if command.start_line > 0 or command.end_line > 0:
            start = command.start_line or 1 
            end   = command.end_line   or "$"
            cmd = f"sed -n '{start},{end}p' {command.path}"
        else:
            cmd = f"cat{' -n' if command.number else ''} {command.path}"
        return VMResponseFormatter._format_command(cmd, response.content)
    
    @staticmethod
    def format_search_response(command: RequestSearchVMCommand, response: ListResponse) -> str:
        root    = shlex.quote(command.root)
        pattern = shlex.quote(command.pattern)
        cmd = f"rg -n --no-heading -e {pattern} {root}"
        body = "\n".join(
            f"{m.path}:{m.line}:{m.line_text}" for m in response.matches
        )
        return VMResponseFormatter._format_command(cmd, body)

    @staticmethod
    def format_find_response(command: RequestFindVMCommand, response: ListResponse) -> str:
        body = "\n".join(f"{entry.name}/" if entry.is_dir else entry.name for entry in response.entries) or "."
        return VMResponseFormatter._format_command(f"find -name {command.name} {command.root}", body)

    @staticmethod
    def _format_command(command: str, response: Message) -> str:
        return f"{command}\n{response}"