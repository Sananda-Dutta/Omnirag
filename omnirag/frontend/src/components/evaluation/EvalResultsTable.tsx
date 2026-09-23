import type { EvalReport } from '@/lib/types'

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : `${Math.round(value * 100)}%`
}

function SummaryStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5 rounded-sm border border-ink-700 bg-ink-850 px-3 py-2">
      <span className="font-mono text-[11px] uppercase tracking-wide text-paper-faint">{label}</span>
      <span className="font-display text-lg text-paper-100">{value}</span>
    </div>
  )
}

export function EvalResultsTable({ report }: { report: EvalReport }) {
  const { summary } = report

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2 font-mono text-xs text-paper-faint">
        <span>scored by</span>
        <span className="rounded-full border border-ink-600 px-2 py-0.5 text-paper-muted">
          {report.eval_provider}
        </span>
        <span>· top_k={report.top_k}</span>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <SummaryStat label="Context found" value={pct(summary.context_found_rate)} />
        <SummaryStat label="Avg. faithfulness" value={pct(summary.avg_faithfulness)} />
        <SummaryStat label="Avg. relevancy" value={pct(summary.avg_answer_relevancy)} />
        <SummaryStat label="Avg. correctness" value={pct(summary.avg_answer_correctness)} />
        <SummaryStat label="Precision@k" value={pct(summary.avg_precision_at_k)} />
        <SummaryStat label="Recall@k" value={pct(summary.avg_recall_at_k)} />
        <SummaryStat label="MRR" value={pct(summary.avg_mrr)} />
        <SummaryStat label="Hit rate" value={pct(summary.avg_hit_rate)} />
      </div>

      <ul className="flex flex-col gap-3">
        {report.results.map((result, i) => (
          <li key={i} className="rounded-sm border border-ink-700 bg-ink-850 p-3">
            <p className="text-sm text-paper-100">{result.question}</p>
            <p className="mt-1.5 whitespace-pre-wrap text-sm text-paper-muted">{result.answer}</p>

            <div className="mt-2.5 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11px] text-paper-faint">
              <span>faithfulness {pct(result.generation.faithfulness)}</span>
              <span>relevancy {pct(result.generation.answer_relevancy)}</span>
              {result.generation.answer_correctness !== null && (
                <span>correctness {pct(result.generation.answer_correctness)}</span>
              )}
              {result.retrieval && (
                <>
                  <span>precision@k {pct(result.retrieval.precision_at_k)}</span>
                  <span>recall@k {pct(result.retrieval.recall_at_k)}</span>
                  <span>mrr {result.retrieval.mrr.toFixed(2)}</span>
                </>
              )}
            </div>

            {result.retrieved_document_filenames.length > 0 && (
              <p className="mt-2 font-mono text-[11px] text-paper-faint">
                sources: {result.retrieved_document_filenames.join(', ')}
              </p>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
