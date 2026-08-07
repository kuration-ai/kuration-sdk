import { ReactWidget } from '@jupyterlab/ui-components';
import * as React from 'react';

import { KurationClient } from '../api/client';
import { IShardEntry, IStatsResponse } from '../api/types';
import { ErrorBox, JsonTree, KeyValueTable, Spinner, useAsync } from '../components/common';

type TabName = 'overview' | 'stats' | 'examples' | 'shards' | 'diff';

function OverviewTab(props: { client: KurationClient; dataset: string }): JSX.Element {
  const { data, error, loading, reload } = useAsync(
    () => props.client.inspect(props.dataset),
    [props.dataset]
  );
  if (loading) {
    return <Spinner label="inspecting…" />;
  }
  if (error) {
    return <ErrorBox error={error} />;
  }
  const inspection = data as Record<string, any>;
  return (
    <div>
      <KeyValueTable
        rows={[
          ['container', String(inspection.container ?? '—')],
          ['shards', String(inspection.shards ?? '—')],
          ['sources', String(inspection.sources ?? '—')],
          ['errors', String(inspection.errors ?? 0)]
        ]}
      />
      <JsonTree value={inspection} label="inspection" />
      <button className="jp-Button" onClick={reload}>
        Reload
      </button>
    </div>
  );
}

function StatsTab(props: { client: KurationClient; dataset: string }): JSX.Element {
  const { client, dataset } = props;
  const { data, error, loading } = useAsync<{
    stats: IStatsResponse;
    html: string;
  }>(async () => {
    const stats = await client.stats(dataset);
    const html = await client.statsHtml(dataset);
    return { stats, html };
  }, [dataset]);
  const [plotUrl, setPlotUrl] = React.useState<string | null>(null);
  const [plotError, setPlotError] = React.useState<unknown>(null);
  React.useEffect(
    () => () => {
      if (plotUrl) {
        URL.revokeObjectURL(plotUrl);
      }
    },
    [plotUrl]
  );
  if (loading) {
    return <Spinner label="computing stats…" />;
  }
  if (error) {
    return <ErrorBox error={error} />;
  }
  const capabilities = data!.stats.capabilities;
  const plots: string[] = capabilities.plots ?? [];
  const showPlot = async (kind: string): Promise<void> => {
    setPlotError(null);
    try {
      const url = await client.statsPlot(dataset, { kind });
      setPlotUrl(previous => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return url;
      });
    } catch (failure) {
      setPlotError(failure);
    }
  };
  return (
    <div>
      {plots.length ? (
        <div className="kur-toolbar">
          {plots.map(kind => (
            <button className="jp-Button" key={kind} onClick={() => void showPlot(kind)}>
              {kind}
            </button>
          ))}
        </div>
      ) : null}
      {plotError ? <ErrorBox error={plotError} /> : null}
      {plotUrl ? <img className="kur-plot" src={plotUrl} alt="stats plot" /> : null}
      {/* Server-rendered, escape-built HTML from render_stats_html (same-origin, trusted). */}
      <div className="kur-stats-html" dangerouslySetInnerHTML={{ __html: data!.html }} />
    </div>
  );
}

function ExamplesTab(props: { client: KurationClient; dataset: string }): JSX.Element {
  const { client, dataset } = props;
  const stats = useAsync(() => client.stats(dataset), [dataset]);
  const [mode, setMode] = React.useState('auto');
  const examples = useAsync(
    () => client.statsExamples(dataset, { mode }),
    [dataset, mode]
  );
  if (stats.loading || examples.loading) {
    return <Spinner label="collecting examples…" />;
  }
  if (stats.error || examples.error) {
    return <ErrorBox error={stats.error ?? examples.error} />;
  }
  const modes: string[] = stats.data!.capabilities.example_modes ?? ['auto'];
  const groups = (examples.data as Record<string, any>).groups ?? {};
  return (
    <div>
      <div className="kur-toolbar">
        <label>
          mode{' '}
          <select value={mode} onChange={event => setMode(event.target.value)}>
            {['auto', ...modes.filter(name => name !== 'auto')].map(name => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
      </div>
      {Object.entries(groups).map(([group, items]) => (
        <div key={group} className="kur-example-group">
          <h4>{group}</h4>
          <div className="kur-example-grid">
            {(items as Record<string, any>[]).map((item, index) => (
              <div key={index} className="kur-example">
                {(item.frames ?? []).map((frame: string, frameIndex: number) => (
                  <img key={frameIndex} src={frame} alt={`frame ${frameIndex}`} />
                ))}
                <JsonTree value={item} label={`example ${index}`} />
              </div>
            ))}
          </div>
        </div>
      ))}
      {!Object.keys(groups).length ? (
        <JsonTree value={examples.data} label="examples" />
      ) : null}
    </div>
  );
}

function ShardsTab(props: { client: KurationClient; dataset: string }): JSX.Element {
  const { client, dataset } = props;
  const { data, error, loading } = useAsync(() => client.shards(dataset), [dataset]);
  const [verify, setVerify] = React.useState(false);
  const [downloadError, setDownloadError] = React.useState<unknown>(null);
  if (loading) {
    return <Spinner label="listing shards…" />;
  }
  if (error) {
    return <ErrorBox error={error} />;
  }
  const download = async (entry: IShardEntry): Promise<void> => {
    setDownloadError(null);
    try {
      const blob = await client.shardArrays(dataset, entry, { verify });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${entry.source_id}-${entry.shard_index ?? 0}.npz`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (failure) {
      setDownloadError(failure);
    }
  };
  return (
    <div>
      <div className="kur-toolbar">
        <label>
          <input
            type="checkbox"
            checked={verify}
            onChange={event => setVerify(event.target.checked)}
          />{' '}
          verify checksums on read
        </label>
        <span className="kur-muted">container: {data!.container}</span>
      </div>
      {downloadError ? <ErrorBox error={downloadError} /> : null}
      <table className="kur-table">
        <thead>
          <tr>
            <th>source</th>
            <th>entity</th>
            <th>split</th>
            <th>steps</th>
            <th>bytes</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {data!.entries.map((entry, index) => (
            <tr key={index}>
              <td>{entry.source_id}</td>
              <td>{entry.entity_id ?? '—'}</td>
              <td>{entry.split ?? '—'}</td>
              <td>{entry.n_steps ?? '—'}</td>
              <td>{entry.size_bytes ?? '—'}</td>
              <td>
                <button className="jp-Button" onClick={() => void download(entry)}>
                  npz
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DiffTab(props: { client: KurationClient; dataset: string }): JSX.Element {
  const { client, dataset } = props;
  const [other, setOther] = React.useState('');
  const [result, setResult] = React.useState<Record<string, any> | null>(null);
  const [error, setError] = React.useState<unknown>(null);
  const compare = async (): Promise<void> => {
    setError(null);
    setResult(null);
    try {
      setResult(await client.compare(dataset, other));
    } catch (failure) {
      setError(failure);
    }
  };
  return (
    <div>
      <div className="kur-toolbar">
        <input
          className="jp-mod-styled"
          placeholder="path to the other dataset"
          value={other}
          onChange={event => setOther(event.target.value)}
        />
        <button className="jp-Button" disabled={!other} onClick={() => void compare()}>
          Compare
        </button>
      </div>
      {error ? <ErrorBox error={error} /> : null}
      {result ? <JsonTree value={result} label="comparison" /> : null}
    </div>
  );
}

function InspectorView(props: { client: KurationClient; dataset: string }): JSX.Element {
  const [tab, setTab] = React.useState<TabName>('overview');
  const tabs: TabName[] = ['overview', 'stats', 'examples', 'shards', 'diff'];
  return (
    <div className="kur-panel kur-inspector">
      <div className="kur-tabbar">
        {tabs.map(name => (
          <button
            key={name}
            className={`kur-tab${tab === name ? ' kur-tab-active' : ''}`}
            onClick={() => setTab(name)}
          >
            {name}
          </button>
        ))}
      </div>
      <div className="kur-muted kur-subject">{props.dataset}</div>
      {tab === 'overview' ? <OverviewTab {...props} /> : null}
      {tab === 'stats' ? <StatsTab {...props} /> : null}
      {tab === 'examples' ? <ExamplesTab {...props} /> : null}
      {tab === 'shards' ? <ShardsTab {...props} /> : null}
      {tab === 'diff' ? <DiffTab {...props} /> : null}
    </div>
  );
}

export class DatasetInspectorWidget extends ReactWidget {
  constructor(client: KurationClient, dataset: string) {
    super();
    this._client = client;
    this._dataset = dataset;
    this.addClass('kur-widget');
    this.title.label = `Dataset: ${dataset.split('/').pop() ?? dataset}`;
    this.title.iconClass = 'jp-MaterialIcon jp-FolderIcon';
    this.title.closable = true;
  }

  get datasetPath(): string {
    return this._dataset;
  }

  protected render(): JSX.Element {
    return <InspectorView client={this._client} dataset={this._dataset} />;
  }

  private readonly _client: KurationClient;
  private readonly _dataset: string;
}
