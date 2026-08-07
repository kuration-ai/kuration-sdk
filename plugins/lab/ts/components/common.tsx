import * as React from 'react';

import { KurationApiError } from '../api/client';

/** Collapsible JSON tree — the honest v1 rendering for structured SDK payloads. */
export function JsonTree(props: { value: unknown; label?: string }): JSX.Element {
  const { value, label } = props;
  if (value === null || value === undefined) {
    return (
      <span className="kur-json-scalar">
        {label ? `${label}: ` : ''}
        {String(value)}
      </span>
    );
  }
  if (typeof value !== 'object') {
    return (
      <span className="kur-json-scalar">
        {label ? `${label}: ` : ''}
        {typeof value === 'string' ? value : JSON.stringify(value)}
      </span>
    );
  }
  const entries = Array.isArray(value)
    ? value.map((item, index) => [String(index), item] as const)
    : Object.entries(value as Record<string, unknown>);
  const summary = Array.isArray(value)
    ? `${label ?? ''}[${entries.length}]`
    : `${label ?? ''}{${entries.length}}`;
  return (
    <details className="kur-json-node">
      <summary>{summary}</summary>
      <div className="kur-json-children">
        {entries.map(([key, child]) => (
          <div key={key}>
            <JsonTree value={child} label={key} />
          </div>
        ))}
      </div>
    </details>
  );
}

export function KeyValueTable(props: {
  rows: [string, React.ReactNode][];
}): JSX.Element {
  return (
    <table className="kur-kv-table">
      <tbody>
        {props.rows.map(([key, node]) => (
          <tr key={key}>
            <th>{key}</th>
            <td>{node}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ErrorBox(props: { error: unknown }): JSX.Element {
  const { error } = props;
  if (error instanceof KurationApiError) {
    return (
      <div className="kur-error">
        <strong>
          {error.type}: {error.message}
        </strong>
        {error.hint ? <div className="kur-error-hint">hint: {error.hint}</div> : null}
      </div>
    );
  }
  return <div className="kur-error">{String(error)}</div>;
}

export function Spinner(props: { label?: string }): JSX.Element {
  return <div className="kur-spinner">{props.label ?? 'working…'}</div>;
}

export function StateBadge(props: { state: string }): JSX.Element {
  return <span className={`kur-badge kur-badge-${props.state}`}>{props.state}</span>;
}

/** Hook: run an async loader once per dependency change, with error surface. */
export function useAsync<T>(
  loader: () => Promise<T>,
  deps: React.DependencyList
): { data: T | null; error: unknown; loading: boolean; reload: () => void } {
  const [data, setData] = React.useState<T | null>(null);
  const [error, setError] = React.useState<unknown>(null);
  const [loading, setLoading] = React.useState(true);
  const [nonce, setNonce] = React.useState(0);
  React.useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    loader().then(
      result => {
        if (live) {
          setData(result);
          setLoading(false);
        }
      },
      failure => {
        if (live) {
          setError(failure);
          setLoading(false);
        }
      }
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);
  return { data, error, loading, reload: () => setNonce(n => n + 1) };
}
