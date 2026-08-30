import {
  ILayoutRestorer,
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import {
  ICommandPalette,
  MainAreaWidget,
  Notification,
  WidgetTracker
} from '@jupyterlab/apputils';
import { IFileBrowserFactory } from '@jupyterlab/filebrowser';
import { ILauncher } from '@jupyterlab/launcher';

import { KurationClient } from './api/client';
import { addCommands } from './commands';
import { RunRegistryModel } from './models/runRegistry';
import { CommandIDs, IKurationClient, IKurationRunRegistry } from './tokens';
import { DatasetInspectorWidget } from './widgets/datasetInspector';
import { JobPanelWidget } from './widgets/jobPanel';
import { RunMonitorWidget } from './widgets/runMonitor';
import { KurationSidebar } from './widgets/sidebar';

const SUPPORTED_API = 1;

/** Provides the API client and performs the version handshake. */
const clientPlugin: JupyterFrontEndPlugin<KurationClient> = {
  id: 'kuration-sdk-lab:client',
  autoStart: true,
  provides: IKurationClient,
  activate: (_app: JupyterFrontEnd): KurationClient => {
    const client = new KurationClient();
    void client
      .version()
      .then(version => {
        if (version.api !== SUPPORTED_API) {
          Notification.warning(
            `kuration-sdk-lab expects API ${SUPPORTED_API} but the server speaks ` +
              `${version.api}; upgrade with: pip install -U kuration-sdk-lab`,
            { autoClose: false }
          );
        }
      })
      .catch(() => {
        Notification.error(
          'The kuration_sdk_lab server extension is not responding. ' +
            'Install/enable it, then restart the Jupyter server: ' +
            'pip install kuration-sdk-lab',
          { autoClose: false }
        );
      });
    return client;
  }
};

/** Provides the shared run registry model (list polling + live event streams). */
const registryPlugin: JupyterFrontEndPlugin<RunRegistryModel> = {
  id: 'kuration-sdk-lab:run-registry',
  autoStart: true,
  requires: [IKurationClient],
  provides: IKurationRunRegistry,
  activate: (_app: JupyterFrontEnd, client: KurationClient): RunRegistryModel => {
    const registry = new RunRegistryModel(client);
    registry.startPolling();
    return registry;
  }
};

/** Commands, widget trackers, launcher/palette/context-menu entries, file type. */
const commandsPlugin: JupyterFrontEndPlugin<void> = {
  id: 'kuration-sdk-lab:commands',
  autoStart: true,
  requires: [IKurationClient, IKurationRunRegistry],
  optional: [ICommandPalette, ILauncher, ILayoutRestorer, IFileBrowserFactory],
  activate: (
    app: JupyterFrontEnd,
    client: KurationClient,
    registry: RunRegistryModel,
    palette: ICommandPalette | null,
    launcher: ILauncher | null,
    restorer: ILayoutRestorer | null,
    fileBrowserFactory: IFileBrowserFactory | null
  ): void => {
    app.docRegistry.addFileType({
      name: 'kuration-job',
      displayName: 'Kuration Job',
      extensions: ['.kur.json'],
      mimeTypes: ['application/json'],
      contentType: 'file',
      fileFormat: 'text'
    });

    const runTracker = new WidgetTracker<MainAreaWidget<RunMonitorWidget>>({
      namespace: 'kuration-run'
    });
    const inspectorTracker = new WidgetTracker<MainAreaWidget<DatasetInspectorWidget>>({
      namespace: 'kuration-dataset'
    });
    const jobPanelTracker = new WidgetTracker<MainAreaWidget<JobPanelWidget>>({
      namespace: 'kuration-job'
    });

    if (restorer) {
      void restorer.restore(runTracker, {
        command: CommandIDs.openRun,
        args: widget => ({ runId: widget.content.id.replace('kuration-run-', '') }),
        name: widget => widget.content.id
      });
      void restorer.restore(inspectorTracker, {
        command: CommandIDs.inspectDataset,
        args: widget => ({ path: widget.content.datasetPath }),
        name: widget => widget.content.datasetPath
      });
      void restorer.restore(jobPanelTracker, {
        command: CommandIDs.openJobPanel,
        args: widget => ({ path: widget.content.path }),
        name: widget => widget.content.path
      });
    }

    addCommands({
      app,
      client,
      registry,
      fileBrowserFactory,
      runTracker,
      inspectorTracker,
      jobPanelTracker
    });

    const category = 'Kuration';
    if (palette) {
      for (const command of [
        CommandIDs.newJob,
        CommandIDs.openJobPanel,
        CommandIDs.inspectDataset
      ]) {
        palette.addItem({ command, category });
      }
    }
    if (launcher) {
      launcher.add({ command: CommandIDs.newJob, category, rank: 1 });
    }
  }
};

/** Left-area sidebar: runs list + entry points. */
const sidebarPlugin: JupyterFrontEndPlugin<void> = {
  id: 'kuration-sdk-lab:sidebar',
  autoStart: true,
  requires: [IKurationRunRegistry],
  activate: (app: JupyterFrontEnd, registry: RunRegistryModel): void => {
    const sidebar = new KurationSidebar({
      registry,
      onOpenRun: runId => void app.commands.execute(CommandIDs.openRun, { runId }),
      onNewJob: () => void app.commands.execute(CommandIDs.newJob),
      onInspect: () => void app.commands.execute(CommandIDs.inspectDataset)
    });
    app.shell.add(sidebar, 'left', { rank: 600 });
  }
};

export default [clientPlugin, registryPlugin, commandsPlugin, sidebarPlugin];
export { KurationClient } from './api/client';
export { IKurationClient, IKurationRunRegistry } from './tokens';
