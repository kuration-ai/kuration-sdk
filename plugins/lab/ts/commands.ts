import { JupyterFrontEnd } from '@jupyterlab/application';
import { InputDialog, MainAreaWidget, Notification, WidgetTracker } from '@jupyterlab/apputils';
import { IFileBrowserFactory } from '@jupyterlab/filebrowser';

import { KurationClient } from './api/client';
import { RunRegistryModel } from './models/runRegistry';
import { CommandIDs } from './tokens';
import { DatasetInspectorWidget } from './widgets/datasetInspector';
import { JobPanelWidget } from './widgets/jobPanel';
import { RunMonitorWidget } from './widgets/runMonitor';

const JOB_TEMPLATE = {
  job_schema_version: '1.0',
  sources: [],
  adapter: { type: 'timeseries' },
  segmenter: { window: 64, stride: 32 },
  strategy: {
    name: 'edit-me',
    define: {},
    binning: { axes: [] },
    selection: { method: 'uniform-bins' }
  },
  output_preset: { name: 'numpy-timeseries' },
  train_hours: 1.0,
  test_hours: 0.0,
  out: 'curated'
};

export interface ICommandsOptions {
  app: JupyterFrontEnd;
  client: KurationClient;
  registry: RunRegistryModel;
  fileBrowserFactory: IFileBrowserFactory | null;
  runTracker: WidgetTracker<MainAreaWidget<RunMonitorWidget>>;
  inspectorTracker: WidgetTracker<MainAreaWidget<DatasetInspectorWidget>>;
  jobPanelTracker: WidgetTracker<MainAreaWidget<JobPanelWidget>>;
}

export function addCommands(options: ICommandsOptions): void {
  const { app, client, registry, fileBrowserFactory } = options;
  const { commands } = app;

  const selectedBrowserPath = (): string | null => {
    const browser = fileBrowserFactory?.tracker.currentWidget;
    if (!browser) {
      return null;
    }
    const item = browser.selectedItems().next();
    return item.done ? null : item.value.path;
  };

  const openRun = (runId: string): void => {
    const existing = options.runTracker.find(
      widget => widget.content.id === `kuration-run-${runId}`
    );
    if (existing) {
      app.shell.activateById(existing.id);
      return;
    }
    const model = registry.get(runId);
    if (!model) {
      void registry.refresh().then(() => {
        if (registry.get(runId)) {
          openRun(runId);
        } else {
          Notification.warning(`Unknown kuration run: ${runId}`);
        }
      });
      return;
    }
    const content = new RunMonitorWidget(model, client);
    content.id = `kuration-run-${runId}`;
    const widget = new MainAreaWidget({ content });
    widget.id = `kuration-run-main-${runId}`;
    void options.runTracker.add(widget);
    app.shell.add(widget, 'main');
  };

  const openInspector = (dataset: string): void => {
    const existing = options.inspectorTracker.find(
      widget => widget.content.datasetPath === dataset
    );
    if (existing) {
      app.shell.activateById(existing.id);
      return;
    }
    const content = new DatasetInspectorWidget(client, dataset);
    const widget = new MainAreaWidget({ content });
    widget.id = `kuration-dataset-${dataset}`;
    void options.inspectorTracker.add(widget);
    app.shell.add(widget, 'main');
  };

  const openJobPanel = (path: string): void => {
    const existing = options.jobPanelTracker.find(
      widget => widget.content.path === path
    );
    if (existing) {
      app.shell.activateById(existing.id);
      return;
    }
    const content = new JobPanelWidget({
      client,
      registry,
      contents: app.serviceManager.contents,
      path,
      onOpenRun: openRun
    });
    const widget = new MainAreaWidget({ content });
    widget.id = `kuration-job-${path}`;
    void options.jobPanelTracker.add(widget);
    app.shell.add(widget, 'main', { mode: 'split-right' });
  };

  commands.addCommand(CommandIDs.newJob, {
    label: 'New Kuration Job',
    caption: 'Create a .kur.json job document and open it with its action panel',
    execute: async () => {
      const cwd = fileBrowserFactory?.tracker.currentWidget?.model.path ?? '';
      const untitled = await app.serviceManager.contents.newUntitled({
        path: cwd,
        type: 'file',
        ext: '.kur.json'
      });
      await app.serviceManager.contents.save(untitled.path, {
        type: 'file',
        format: 'text',
        content: JSON.stringify(JOB_TEMPLATE, null, 2)
      });
      await commands.execute('docmanager:open', { path: untitled.path });
      openJobPanel(untitled.path);
    }
  });

  commands.addCommand(CommandIDs.openJobPanel, {
    label: 'Open Kuration Job Panel',
    execute: async args => {
      const path = (args.path as string) ?? selectedBrowserPath();
      if (!path) {
        Notification.warning('Select a .kur.json file first.');
        return;
      }
      openJobPanel(path);
    }
  });

  commands.addCommand(CommandIDs.runJob, {
    label: 'Run Kuration Job',
    execute: async args => {
      const path = (args.path as string) ?? selectedBrowserPath();
      if (!path) {
        Notification.warning('Select a .kur.json file first.');
        return;
      }
      openJobPanel(path);
    }
  });

  commands.addCommand(CommandIDs.openRun, {
    label: 'Open Kuration Run',
    execute: args => {
      const runId = args.runId as string;
      if (runId) {
        openRun(runId);
      }
    }
  });

  commands.addCommand(CommandIDs.cancelRun, {
    label: 'Cancel Kuration Run',
    execute: args => {
      const runId = args.runId as string;
      registry.get(runId)?.cancel();
    }
  });

  commands.addCommand(CommandIDs.inspectDataset, {
    label: 'Inspect Kuration Dataset…',
    execute: async args => {
      let path = args.path as string | undefined;
      if (!path) {
        const answer = await InputDialog.getText({
          title: 'Inspect Kuration dataset',
          placeholder: 'path to a written dataset directory'
        });
        if (!answer.button.accept || !answer.value) {
          return;
        }
        path = answer.value;
      }
      openInspector(path);
    }
  });

  app.contextMenu.addItem({
    command: CommandIDs.openJobPanel,
    selector: '.jp-DirListing-item[data-file-type="kuration-job"]',
    rank: 10
  });
}
