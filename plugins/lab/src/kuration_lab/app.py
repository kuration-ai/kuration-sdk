"""KurationLabApp — the jupyter_server extension application.

Configuration (e.g. in ``jupyter_server_config.py``):

    c.KurationLabApp.runs_dir = "/data/kuration-runs"
    c.KurationLabApp.max_concurrent_runs = 2
    c.KurationLabApp.allow_absolute_paths = True
    c.KurationLabApp.cancel_grace_seconds = 15.0
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from jupyter_server.extension.application import ExtensionApp
from traitlets import Bool, Float, Int, Unicode

from . import __version__
from .handlers import handlers
from .jobs.manager import JobManager, default_runs_dir
from .jobs.registry import RunRegistry


class KurationLabApp(ExtensionApp):
    name = "kuration_lab"
    version = __version__
    load_other_extensions = True

    runs_dir = Unicode(
        "", config=True,
        help="Directory holding run state (default: <jupyter data dir>/kuration-lab/runs).")
    max_concurrent_runs = Int(
        2, config=True, help="Worker subprocesses allowed to run at once; excess runs queue.")
    allow_absolute_paths = Bool(
        True, config=True,
        help="Accept absolute filesystem paths in API requests (datasets often live outside "
             "the notebook root). When False, every path must stay under the server root.")
    cancel_grace_seconds = Float(
        15.0, config=True, help="Seconds between SIGTERM and SIGKILL when cancelling a run.")

    def initialize_settings(self):
        registry = RunRegistry(self.runs_dir or default_runs_dir())
        manager = JobManager(registry, max_concurrent=self.max_concurrent_runs,
                             cancel_grace=self.cancel_grace_seconds)
        self.settings["kuration_lab"] = {
            "job_manager": manager,
            "executor": ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="kuration-lab"),
            "root_dir": self.serverapp.root_dir,
            "allow_absolute_paths": self.allow_absolute_paths,
        }

    def initialize_handlers(self):
        self.handlers.extend(handlers)

    async def stop_extension(self):
        lab = self.settings.get("kuration_lab") or {}
        manager = lab.get("job_manager")
        if manager is not None:
            await manager.shutdown()
        executor = lab.get("executor")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
