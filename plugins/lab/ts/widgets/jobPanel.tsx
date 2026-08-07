import { Contents } from '@jupyterlab/services';
import { ReactWidget } from '@jupyterlab/ui-components';
import * as React from 'react';

import { KurationClient } from '../api/client';
import { JobDocument } from '../api/types';
import { ErrorBox, JsonTree, Spinner } from '../components/common';
import { RunRegistryModel } from '../models/runRegistry';

type ActionName = 'validate' | 'explain' | 'preview' | 'run';

interface IActionState {
  action: ActionName;
  result?: unknown;
  error?: unknown;
  running: boolean;
}

function dirname(path: string): string {
  const index = path.lastIndexOf('/');
  return index < 0 ? '' : path.slice(0, index);
}

/**
 * Companion panel for a .kur.json document: the file itself is edited in the stock JSON
 * editor; this panel re-reads the saved file for each action, so what runs is what's on disk.
 */
function JobPanelView(props: {
  client: KurationClient;
  registry: RunRegistryModel;
  contents: Contents.IManager;
  path: string;
  onOpenRun: (runId: string) => void;
}): JSX.Element {
  const { client, registry, contents, path, onOpenRun } = props;
  const [state, setState] = React.useState<IActionState | null>(null);

  const readJob = async (): Promise<JobDocument> => {
    const model = await contents.get(path, { content: true, format: 'text' });
    const raw = model.content as string;
    let job: unknown;
    try {
      job = JSON.parse(raw);
    } catch (failure) {
      throw new Error(`${path} is not valid JSON: ${failure}`);
    }
    if (typeof job !== 'object' || job === null || Array.isArray(job)) {
      throw new Error(`${path} must contain a JSON object`);
    }
    return job as JobDocument;
  };

  const act = async (action: ActionName): Promise<void> => {
    setState({ action, running: true });
    try {
      const job = await readJob();
      const jobDir = dirname(path);
      let result: unknown;
      switch (action) {
        case 'validate':
          result = await client.validateJob(job, jobDir);
          break;
        case 'explain':
          result = await client.explainJob(job, jobDir);
          break;
        case 'preview':
          result = await client.preview(job, jobDir);
          break;
        case 'run': {
          const model = await registry.submit(job, { jobDir });
          result = model.info;
          onOpenRun(model.info.run_id);
          break;
        }
      }
      setState({ action, result, running: false });
    } catch (failure) {
      setState({ action, error: failure, running: false });
    }
  };

  const actions: ActionName[] = ['validate', 'explain', 'preview', 'run'];
  return (
    <div className="kur-panel kur-job-panel">
      <div className="kur-muted kur-subject">{path}</div>
      <div className="kur-toolbar">
        {actions.map(action => (
          <button
            key={action}
            className="jp-Button"
            disabled={state?.running ?? false}
            onClick={() => void act(action)}
          >
            {action === 'run' ? 'Run ▶' : action[0].toUpperCase() + action.slice(1)}
          </button>
        ))}
      </div>
      {state?.running ? <Spinner label={`${state.action}…`} /> : null}
      {state?.error ? <ErrorBox error={state.error} /> : null}
      {state?.result !== undefined && !state.running ? (
        <JsonTree value={state.result} label={state.action} />
      ) : null}
      {!state ? (
        <div className="kur-muted">
          Edit the job in the JSON editor, save, then Validate / Explain / Preview / Run.
        </div>
      ) : null}
    </div>
  );
}

export class JobPanelWidget extends ReactWidget {
  constructor(options: {
    client: KurationClient;
    registry: RunRegistryModel;
    contents: Contents.IManager;
    path: string;
    onOpenRun: (runId: string) => void;
  }) {
    super();
    this._options = options;
    this.addClass('kur-widget');
    this.title.label = `Job: ${options.path.split('/').pop() ?? options.path}`;
    this.title.iconClass = 'jp-MaterialIcon jp-SettingsIcon';
    this.title.closable = true;
  }

  get path(): string {
    return this._options.path;
  }

  protected render(): JSX.Element {
    return <JobPanelView {...this._options} />;
  }

  private readonly _options: {
    client: KurationClient;
    registry: RunRegistryModel;
    contents: Contents.IManager;
    path: string;
    onOpenRun: (runId: string) => void;
  };
}
