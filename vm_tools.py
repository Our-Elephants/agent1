from typing import Literal

from haystack.tools import tool, Toolset

from vm_api import (
    VM, VMResponseFormatter,
    RequestTreeVMCommand, RequestFindVMCommand, RequestSearchVMCommand, RequestListVMCommand,
    RequestReadVMCommand, RequestContextVMCommand, RequestWriteVMCommand,
    RequestDeleteVMCommand, RequestMkDirVMCommand, RequestMoveVMCommand,
    RequestReportTaskCompletionVMCommand, OutcomeEnum,
)

_current_vm: VM | None = None


@tool(description="Show directory tree (args: level, root)")
def tree(level: int = 2, root: str = "/") -> str:
    cmd = RequestTreeVMCommand(level=level, root=root)
    resp = _current_vm.execute_tree_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Find files or directories by name pattern. limit: max 20 results.")
def find(name: str, kind: Literal["all", "files", "dirs"] = "all", limit: int = 10, root: str = "/") -> str:
    cmd = RequestFindVMCommand(name=name, kind=kind, limit=limit, root=root)
    resp = _current_vm.execute_find_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Search file contents by regex pattern. limit: max 20 results.")
def search(pattern: str, limit: int = 10, root: str = "/") -> str:
    cmd = RequestSearchVMCommand(pattern=pattern, limit=limit, root=root)
    resp = _current_vm.execute_search_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="List directory contents (args: path)")
def ls(path: str = "/") -> str:
    cmd = RequestListVMCommand(path=path)
    resp = _current_vm.execute_list_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Read file contents. start_line/end_line are 1-based inclusive (0 = default = full file). Set number=true to show line numbers.")
def read(path: str, number: bool = False, start_line: int = 0, end_line: int = 0) -> str:
    cmd = RequestReadVMCommand(path=path, number=number, start_line=start_line, end_line=end_line)
    resp = _current_vm.execute_read_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Get VM runtime context (current time). Takes no arguments.")
def context() -> str:
    cmd = RequestContextVMCommand()
    resp = _current_vm.execute_context_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Write or create a file. Omit start_line/end_line (or set to 0) for full-file write. For partial edits: start_line and end_line are 1-based inclusive line numbers — the content replaces lines start_line through end_line.")
def write(path: str, content: str, start_line: int = 0, end_line: int = 0) -> str:
    cmd = RequestWriteVMCommand(path=path, content=content, start_line=start_line, end_line=end_line)
    resp = _current_vm.execute_write_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Delete a file at the given path. To delete a directory, delete each file inside it first.")
def delete(path: str) -> str:
    cmd = RequestDeleteVMCommand(path=path)
    resp = _current_vm.execute_delete_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Create directory (args: path)")
def mkdir(path: str) -> str:
    cmd = RequestMkDirVMCommand(path=path)
    resp = _current_vm.execute_mkdir_command(cmd)
    return VMResponseFormatter.format(cmd, resp)


@tool(description="Move or rename file/directory (args: from_name, to_name)")
def move(from_name: str, to_name: str) -> str:
    cmd = RequestMoveVMCommand(from_name=from_name, to_name=to_name)
    resp = _current_vm.execute_move_command(cmd)
    return VMResponseFormatter.format(cmd, resp)

@tool(description=(
    "Report task completion. Call this when done or blocked. "
    "outcome must be one of: OUTCOME_OK, OUTCOME_DENIED_SECURITY, "
    "OUTCOME_NONE_CLARIFICATION, OUTCOME_NONE_UNSUPPORTED, OUTCOME_ERR_INTERNAL."
))
def report_task_completion(message: str, outcome: str, grounding_refs: list[str] | None = None) -> str:
    return "noop"

_VM_TOOLSET = Toolset([tree, find, search, ls, read, context, write, delete, mkdir, move, report_task_completion])

def make_toolset(vm: VM) -> Toolset:
    global _current_vm
    _current_vm = vm
    return _VM_TOOLSET
