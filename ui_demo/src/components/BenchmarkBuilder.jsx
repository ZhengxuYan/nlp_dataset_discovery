import { useEffect, useMemo, useState } from 'react';
import {
  BookOpen,
  Bot,
  ChevronLeft,
  ChevronRight,
  Database,
  FileSearch,
  Link as LinkIcon,
  RefreshCcw,
  Save,
  Upload,
  WandSparkles,
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:8123';

const ROW_JOBS = [
  { id: 'run_pipeline_for_query', label: 'Run Pipeline', icon: WandSparkles },
  { id: 'extract_candidates', label: 'Extract', icon: Bot },
  { id: 'resolve_candidates', label: 'Resolve', icon: LinkIcon },
  { id: 'fetch_papers', label: 'Fetch', icon: BookOpen },
  { id: 'process_papers', label: 'Process', icon: Database },
  { id: 'build_drafts', label: 'Rebuild Draft', icon: RefreshCcw },
];

const BULK_JOBS = [
  { id: 'bulk_extract_missing', label: 'Extract Missing' },
  { id: 'bulk_resolve_all', label: 'Resolve All' },
  { id: 'bulk_fetch_all', label: 'Fetch All' },
  { id: 'bulk_process_all', label: 'Process All' },
  { id: 'bulk_build_all', label: 'Rebuild All Drafts' },
];

const STATUS_TONE = {
  complete: 'success',
  needs_annotation: 'warning',
  needs_processing: 'warning',
  needs_fetch: 'warning',
  needs_resolution: 'warning',
  needs_extraction: 'warning',
};

const FILTERS = [
  { id: 'all', label: 'All Rows' },
  { id: 'ready', label: 'Ready for Review' },
  { id: 'attention', label: 'Needs My Attention' },
];

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function Badge({ status }) {
  const tone = STATUS_TONE[status] || 'warning';
  return <span className={`badge ${tone}`}>{status}</span>;
}

function JobBadge({ status }) {
  if (!status) return null;
  return <span className={`badge ${status === 'running' ? 'warning' : 'success'}`}>{status}</span>;
}

function BenchmarkBuilder() {
  const [queries, setQueries] = useState([]);
  const [counts, setCounts] = useState({});
  const [jobs, setJobs] = useState([]);
  const [selectedPaperId, setSelectedPaperId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [jobOutput, setJobOutput] = useState('');
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [refreshingDetail, setRefreshingDetail] = useState(false);
  const [saving, setSaving] = useState(false);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [manualPaperForm, setManualPaperForm] = useState({});
  const [uploadingCandidateId, setUploadingCandidateId] = useState(null);

  async function loadQueries() {
    const data = await fetchJson(`${API_BASE}/api/queries`);
    setQueries(data.items || []);
    setCounts(data.counts || {});
    if (!selectedPaperId && data.items?.length) {
      setSelectedPaperId(data.items[0].paper_id);
    }
  }

  async function loadJobs() {
    const data = await fetchJson(`${API_BASE}/api/jobs`);
    setJobs(data.items || []);
  }

  async function loadDetail(paperId, { silent = false } = {}) {
    if (!paperId) return;
    if (silent) setRefreshingDetail(true);
    else setLoadingDetail(true);
    try {
      const data = await fetchJson(`${API_BASE}/api/query/${encodeURIComponent(paperId)}`);
      setDetail(data);
    } finally {
      if (silent) setRefreshingDetail(false);
      else setLoadingDetail(false);
    }
  }

  useEffect(() => {
    loadQueries();
    loadJobs();
    const interval = setInterval(() => {
      loadQueries();
      loadJobs();
      if (selectedPaperId) {
        loadDetail(selectedPaperId, { silent: true });
      }
    }, 4000);
    return () => clearInterval(interval);
  }, [selectedPaperId]);

  useEffect(() => {
    loadDetail(selectedPaperId);
  }, [selectedPaperId]);

  const filteredQueries = useMemo(() => {
    const matchesSearch = (query) => {
      const normalized = searchQuery.trim().toLowerCase();
      if (!normalized) return true;
      const haystack = [
        query.paper_id,
        query.title,
        query.arxiv_id,
        ...(query.query_dataset_names || []),
        ...(query.priority_task_names || []),
      ].filter(Boolean).join(' ').toLowerCase();
      return haystack.includes(normalized);
    };
    let baseQueries = queries;
    if (filter === 'ready') {
      baseQueries = queries.filter((query) => query.status === 'needs_annotation');
      return baseQueries.filter(matchesSearch);
    }
    if (filter === 'attention') {
      const priorityRows = queries.filter((query) => query.priority_task_count > 0 && query.status !== 'complete');
      baseQueries = priorityRows.length ? priorityRows : queries.filter((query) => query.job_status === 'failed');
      return baseQueries.filter(matchesSearch);
    }
    return baseQueries.filter(matchesSearch);
  }, [queries, filter, searchQuery]);

  const currentIndex = useMemo(
    () => filteredQueries.findIndex((query) => query.paper_id === selectedPaperId),
    [filteredQueries, selectedPaperId],
  );

  const draft = detail?.draft;
  const candidates = detail?.candidates || [];
  const priorityTasks = detail?.priority_tasks || [];
  const priorityTaskByCandidateId = useMemo(() => {
    const mapping = new Map();
    priorityTasks.forEach((task) => mapping.set(task.candidate_id, task));
    return mapping;
  }, [priorityTasks]);
  const linkedPriorPaperIds = useMemo(() => new Set(draft?.gold_prior_paper_ids || []), [draft]);
  const hardNegativeIds = useMemo(() => new Set(draft?.hard_negative_ids || []), [draft]);
  const softNegativeIds = useMemo(() => new Set(draft?.soft_negative_ids || []), [draft]);

  const sortedCandidates = useMemo(() => (
    [...candidates].sort((left, right) => {
      const resolutionRank = (candidate) => (
        candidate.resolution_status === 'resolved_in_db' || candidate.resolution_status === 'resolved_arxiv' ? 0
          : candidate.resolution_status === 'ambiguous' ? 1
            : 2
      );
      const relationshipRank = (candidate) => (
        candidate.relationship_type === 'closest_prior_dataset' ? 0
          : candidate.relationship_type === 'source_dataset' ? 1
            : candidate.relationship_type === 'parallel_benchmark' ? 2
              : candidate.relationship_type === 'evaluation_baseline' ? 3
                : 4
      );
      const confidenceRank = (candidate) => (
        candidate.confidence === 'high' ? 0
          : candidate.confidence === 'medium' ? 1
            : 2
      );
      const priorityRank = (candidate) => (
        priorityTaskByCandidateId.has(candidate.candidate_id) ? 0 : 1
      );
      const displayTitle = (candidate) => (
        candidate.dataset_name || candidate.reference_title || candidate.paper_title || candidate.candidate_id
      );
      return priorityRank(left) - priorityRank(right)
        || resolutionRank(left) - resolutionRank(right)
        || relationshipRank(left) - relationshipRank(right)
        || confidenceRank(left) - confidenceRank(right)
        || displayTitle(left).localeCompare(displayTitle(right));
    })
  ), [candidates, priorityTaskByCandidateId]);

  function applyDraftPatch(localPatch) {
    setDetail((current) => current?.draft ? ({
      ...current,
      draft: { ...current.draft, ...localPatch },
    }) : current);
  }

  async function saveDraftPatch(patch) {
    if (!detail?.draft) return;
    setSaving(true);
    try {
      const updated = await fetchJson(`${API_BASE}/api/draft/${encodeURIComponent(detail.paper_id)}`, {
        method: 'PATCH',
        body: JSON.stringify(patch),
      });
      setDetail((current) => ({ ...current, draft: updated }));
      await loadQueries();
    } finally {
      setSaving(false);
    }
  }

  async function saveCandidatePatch(candidateId, patch) {
    const updated = await fetchJson(`${API_BASE}/api/candidate/${encodeURIComponent(candidateId)}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
    setDetail((current) => ({
      ...current,
      candidates: current.candidates.map((candidate) => (
        candidate.candidate_id === candidateId ? updated : candidate
      )),
    }));
  }

  async function savePaperPatch(paperId, patch) {
    await fetchJson(`${API_BASE}/api/paper/${encodeURIComponent(paperId)}`, {
      method: 'PATCH',
      body: JSON.stringify(patch),
    });
    await loadDetail(selectedPaperId);
  }

  async function runJob(jobId) {
    const result = await fetchJson(`${API_BASE}/api/run-job`, {
      method: 'POST',
      body: JSON.stringify({ job: jobId, paper_id: detail?.paper_id }),
    });
    setJobOutput(`Queued ${result.job_type} (${result.job_id})`);
    await loadJobs();
    await loadQueries();
  }

  async function runBulkJob(jobId) {
    const result = await fetchJson(`${API_BASE}/api/run-bulk-job`, {
      method: 'POST',
      body: JSON.stringify({ job: jobId }),
    });
    setJobOutput(`Queued ${result.job_type} (${result.job_id})`);
    await loadJobs();
  }

  function toggleDraftList(field, value) {
    if (!value) return;
    const current = new Set(draft?.[field] || []);
    if (current.has(value)) current.delete(value);
    else current.add(value);
    const next = Array.from(current);
    applyDraftPatch({ [field]: next });
    saveDraftPatch({ [field]: next });
  }

  async function acceptSuggestions() {
    if (!draft) return;
    const patch = {
      gold_prior_paper_ids: draft.suggested_gold_prior_paper_ids || [],
      hard_negative_ids: draft.suggested_hard_negative_ids || [],
      soft_negative_ids: draft.suggested_soft_negative_ids || [],
    };
    applyDraftPatch(patch);
    await saveDraftPatch(patch);
  }

  async function markComplete() {
    const status = (draft?.gold_prior_paper_ids || []).length > 0 && draft?.gold_added_information_label
      ? 'complete'
      : 'needs_annotation';
    await saveDraftPatch({ draft_status: status });
  }

  async function createManualPaper(candidate) {
    const form = manualPaperForm[candidate.candidate_id] || {};
    const created = await fetchJson(`${API_BASE}/api/create-paper-record`, {
      method: 'POST',
      body: JSON.stringify({
        candidate_id: candidate.candidate_id,
        query_paper_id: detail.paper_id,
        title: form.title || candidate.reference_title || candidate.paper_title || candidate.dataset_name || 'Manual prior paper',
        year: form.year ? Number(form.year) : null,
        authors: form.authors ? form.authors.split(',').map((s) => s.trim()).filter(Boolean) : [],
        notes: form.notes || '',
      }),
    });
    setJobOutput(`Created manual paper record ${created.paper_id}`);
    await loadDetail(selectedPaperId);
    await loadQueries();
  }

  async function uploadPdf(candidate, file) {
    if (!file) return;
    setUploadingCandidateId(candidate.candidate_id);
    try {
      const form = manualPaperForm[candidate.candidate_id] || {};
      const body = new FormData();
      body.append('candidate_id', candidate.candidate_id);
      body.append('query_paper_id', detail.paper_id);
      body.append('title', form.title || candidate.paper_title || candidate.dataset_name || 'Manual prior paper');
      if (candidate.resolved_paper_id) body.append('paper_id', candidate.resolved_paper_id);
      body.append('file', file);
      const response = await fetch(`${API_BASE}/api/upload-pdf`, { method: 'POST', body });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = await response.json();
      setJobOutput(`Uploaded PDF for ${payload.title}`);
      await loadDetail(selectedPaperId);
      await loadQueries();
    } finally {
      setUploadingCandidateId(null);
    }
  }

  async function linkArxiv(candidate) {
    const form = manualPaperForm[candidate.candidate_id] || {};
    if (!form.arxiv?.trim()) {
      setJobOutput('Enter an arXiv URL or ID first.');
      return;
    }
    const payload = await fetchJson(`${API_BASE}/api/link-arxiv`, {
      method: 'POST',
      body: JSON.stringify({
        candidate_id: candidate.candidate_id,
        query_paper_id: detail.paper_id,
        arxiv: form.arxiv,
        title: form.title || candidate.reference_title || candidate.paper_title || candidate.dataset_name || candidate.candidate_id,
        year: form.year ? Number(form.year) : candidate.reference_year || candidate.year || null,
        authors: form.authors ? form.authors.split(',').map((s) => s.trim()).filter(Boolean) : candidate.reference_authors || candidate.authors || [],
        notes: form.notes || '',
        fetch: true,
      }),
    });
    setJobOutput(`Linked ${payload.paper.paper_id}${payload.job ? ` and queued fetch job ${payload.job.job_id}` : ''}`);
    await loadDetail(selectedPaperId);
    await loadQueries();
    await loadJobs();
  }

  function moveSelection(offset) {
    if (currentIndex < 0) return;
    const next = filteredQueries[currentIndex + offset];
    if (next) setSelectedPaperId(next.paper_id);
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1 className="flex-center" style={{ justifyContent: 'flex-start', gap: 10 }}>
            <FileSearch className="gradient-text" size={28} />
            Prior-Work Builder
          </h1>
          <p className="sidebar-subtitle">Queue jobs in the background and review rows one at a time.</p>
        </div>

        <div className="bulk-panel glass-panel">
          <div className="bulk-title">Bulk Actions</div>
          <div className="model-note">LLM model: {counts.llm_model || 'unknown'}</div>
          <div className="counts-grid">
            <div><strong>{counts.needs_extraction || 0}</strong><span>Need extraction</span></div>
            <div><strong>{counts.unresolved_candidates || 0}</strong><span>Unresolved</span></div>
            <div><strong>{counts.waiting_fetch || 0}</strong><span>Waiting fetch</span></div>
            <div><strong>{counts.waiting_processing || 0}</strong><span>Waiting processing</span></div>
            <div><strong>{counts.ready_for_review || 0}</strong><span>Ready review</span></div>
            <div><strong>{counts.complete || 0}</strong><span>Human complete</span></div>
            <div><strong>{counts.priority_attention_rows || 0}</strong><span>Priority rows</span></div>
          </div>
          <div className="bulk-actions">
            {BULK_JOBS.map((job) => (
              <button key={job.id} onClick={() => runBulkJob(job.id)}>{job.label}</button>
            ))}
          </div>
        </div>

        <div className="sidebar-actions">
          <button onClick={() => { loadQueries(); loadJobs(); }}>
            <RefreshCcw size={16} />
            Refresh
          </button>
        </div>

        <div className="filter-row">
          {FILTERS.map((item) => (
            <button
              key={item.id}
              className={filter === item.id ? 'primary compact' : 'compact'}
              onClick={() => setFilter(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>

        <div className="search-box">
          <input
            value={searchQuery}
            onChange={(event) => setSearchQuery(event.target.value)}
            placeholder="Search arXiv ID, paper, dataset, or priority candidate..."
          />
          {searchQuery ? (
            <button className="compact" onClick={() => setSearchQuery('')}>Clear</button>
          ) : null}
        </div>

        <div className="query-list">
          {filteredQueries.map((query) => (
            <div
              key={query.paper_id}
              className={`query-list-item ${selectedPaperId === query.paper_id ? 'selected' : ''}`}
              onClick={() => setSelectedPaperId(query.paper_id)}
            >
              <div className="query-list-title">{query.title}</div>
              <div className="query-list-meta">
                <Badge status={query.status} />
                <JobBadge status={query.job_status} />
                <span>{query.candidate_count} candidates</span>
                {query.priority_task_count ? <span>{query.priority_task_count} priority</span> : null}
              </div>
              {query.priority_task_names?.length ? (
                <div className="query-list-priority">
                  Priority: {query.priority_task_names.slice(0, 2).join(', ')}
                </div>
              ) : null}
              <div className="query-list-datasets">{query.query_dataset_names.join(', ') || 'No introduced dataset'}</div>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-content">
        <div className="bg-glow" />
        <div className="content-wrapper builder-layout">
          {!detail || loadingDetail ? (
            <div className="glass-panel card empty-state">
              <div className="page-title">{loadingDetail ? 'Loading query...' : 'Select a query paper'}</div>
            </div>
          ) : (
            <>
              <section className="page-header builder-header">
                <div>
                  <h2 className="page-title">{detail.title}</h2>
                  <p className="page-description">
                    {detail.arxiv_id ? `arXiv: ${detail.arxiv_id}` : 'No arXiv id'} · <Badge status={detail.status} />
                    {detail.jobs?.length ? <> · <JobBadge status={detail.jobs[0].status} /></> : null}
                    {refreshingDetail ? <> · <span className="refreshing-note">refreshing</span></> : null}
                  </p>
                </div>
                <div className="job-toolbar">
                  {ROW_JOBS.map((job) => {
                    const Icon = job.icon;
                    return (
                      <button key={job.id} onClick={() => runJob(job.id)}>
                        <Icon size={16} />
                        {job.label}
                      </button>
                    );
                  })}
                </div>
              </section>

              <section className="nav-row">
                <button onClick={() => moveSelection(-1)} disabled={currentIndex <= 0}>
                  <ChevronLeft size={16} /> Previous Row
                </button>
                <button onClick={() => moveSelection(1)} disabled={currentIndex < 0 || currentIndex >= filteredQueries.length - 1}>
                  Next Row <ChevronRight size={16} />
                </button>
              </section>

              <section className="glass-panel card section-block">
                <div className="card-title">Query Datasets</div>
                <div className="dataset-grid">
                  {(detail.query_datasets || []).map((dataset) => (
                    <div key={dataset.name} className="dataset-card">
                      <div className="dataset-name">{dataset.name}</div>
                      <div className="dataset-meta">{dataset.domain} · {dataset.role}</div>
                      <ul className="acu-list">
                        {(dataset.acus || []).map((acu) => <li key={acu}>{acu}</li>)}
                      </ul>
                    </div>
                  ))}
                </div>
              </section>

              {priorityTasks.length ? (
                <section className="glass-panel card section-block">
                  <div className="card-title">Priority Candidates To Complete</div>
                  <div className="priority-task-list">
                    {priorityTasks.map((task) => (
                      <div key={task.candidate_id} className="priority-task-card">
                        <div>
                          <strong>#{task.rank}: {task.candidate_name || task.reference_title || task.candidate_id}</strong>
                          <div className="candidate-meta">
                            {task.relationship_type} · {task.action_needed} · {task.confidence}
                          </div>
                          {task.reference_title ? <div><strong>Reference:</strong> {task.reference_title}</div> : null}
                          {task.rationale ? <div className="candidate-description">{task.rationale}</div> : null}
                        </div>
                        {task.reference_url ? (
                          <a href={task.reference_url} target="_blank" rel="noreferrer" className="inline-link">
                            Open URL
                          </a>
                        ) : null}
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}

              <section className="content-split">
                <div className="glass-panel card section-block">
                  <div className="card-title flex-between">
                    <span>Previous-Work Candidates</span>
                    <button className="compact" onClick={acceptSuggestions}>
                      <WandSparkles size={16} />
                      Accept Suggestions
                    </button>
                  </div>
                  <div className="candidate-list">
                    {sortedCandidates.map((candidate) => {
                      const manualForm = manualPaperForm[candidate.candidate_id] || {};
                      const needsManual = ['unresolved', 'ambiguous', 'needs_resolution'].includes(candidate.resolution_status);
                      const isSuggestedGold = (draft?.suggested_gold_prior_paper_ids || []).includes(candidate.resolved_paper_id);
                      const priorityTask = priorityTaskByCandidateId.get(candidate.candidate_id);
                      return (
                        <div key={candidate.candidate_id} className={`candidate-card ${isSuggestedGold || priorityTask ? 'candidate-card--highlight' : ''}`}>
                          <div className="candidate-header">
                            <div>
                              <div className="candidate-title">{candidate.dataset_name || candidate.reference_title || candidate.paper_title || candidate.candidate_id}</div>
                              <div className="candidate-meta">
                                <Badge status={candidate.resolution_status} /> · {candidate.relationship_type} · {candidate.confidence}
                                {candidate.resolution_source ? <> · {candidate.resolution_source}</> : null}
                                {candidate.citation_key ? <> · cite:{candidate.citation_key}</> : null}
                                {priorityTask ? <span className="suggestion-pill">priority #{priorityTask.rank}</span> : null}
                                {isSuggestedGold ? <span className="suggestion-pill">suggested gold</span> : null}
                              </div>
                            </div>
                            <div className="candidate-toggle-group">
                              <label>
                                <input
                                  type="checkbox"
                                  checked={linkedPriorPaperIds.has(candidate.resolved_paper_id)}
                                  disabled={!candidate.resolved_paper_id || !draft}
                                  onChange={() => toggleDraftList('gold_prior_paper_ids', candidate.resolved_paper_id)}
                                />
                                Gold prior
                              </label>
                              <label>
                                <input
                                  type="checkbox"
                                  checked={hardNegativeIds.has(candidate.candidate_id)}
                                  disabled={!draft}
                                  onChange={() => toggleDraftList('hard_negative_ids', candidate.candidate_id)}
                                />
                                Hard negative
                              </label>
                              <label>
                                <input
                                  type="checkbox"
                                  checked={softNegativeIds.has(candidate.candidate_id)}
                                  disabled={!draft}
                                  onChange={() => toggleDraftList('soft_negative_ids', candidate.candidate_id)}
                                />
                                Soft negative
                              </label>
                            </div>
                          </div>
                          <div className="candidate-description">{candidate.description}</div>
                          <div className="candidate-evidence">{candidate.evidence_text}</div>
                          {(candidate.reference_title || candidate.reference_arxiv_id || candidate.reference_url) ? (
                            <div className="reference-panel">
                              <div className="reference-panel-title">Bibliography match</div>
                              {candidate.reference_title ? <div><strong>Title:</strong> {candidate.reference_title}</div> : null}
                              {candidate.reference_authors?.length ? <div><strong>Authors:</strong> {candidate.reference_authors.slice(0, 4).join(', ')}</div> : null}
                              {candidate.reference_year ? <div><strong>Year:</strong> {candidate.reference_year}</div> : null}
                              {candidate.reference_arxiv_id ? <div><strong>arXiv:</strong> {candidate.reference_arxiv_id}</div> : null}
                              {candidate.reference_url ? (
                                <a href={candidate.reference_url} target="_blank" rel="noreferrer" className="inline-link">
                                  Open bibliography URL
                                </a>
                              ) : null}
                            </div>
                          ) : null}
                          <div className="candidate-fields">
                            <input
                              value={candidate.paper_title || ''}
                              onChange={(event) => setDetail((current) => ({
                                ...current,
                                candidates: current.candidates.map((item) => (
                                  item.candidate_id === candidate.candidate_id ? { ...item, paper_title: event.target.value } : item
                                )),
                              }))}
                              onBlur={(event) => saveCandidatePatch(candidate.candidate_id, { paper_title: event.target.value })}
                              placeholder="Paper title"
                            />
                            <textarea
                              value={candidate.annotation_notes || ''}
                              onChange={(event) => setDetail((current) => ({
                                ...current,
                                candidates: current.candidates.map((item) => (
                                  item.candidate_id === candidate.candidate_id ? { ...item, annotation_notes: event.target.value } : item
                                )),
                              }))}
                              onBlur={(event) => saveCandidatePatch(candidate.candidate_id, { annotation_notes: event.target.value })}
                              placeholder="Notes"
                              rows={2}
                            />
                            {candidate.resolved_url && (
                              <a href={candidate.resolved_url} target="_blank" rel="noreferrer" className="inline-link">
                                Open resolved paper
                              </a>
                            )}
                          </div>

                          {candidate.resolved_paper_id ? (
                            <div className="manual-panel">
                              <div className="manual-panel-title">Linked paper metadata</div>
                              <button className="compact" onClick={() => savePaperPatch(candidate.resolved_paper_id, { notes: (manualForm.notes || candidate.annotation_notes || '') })}>
                                Save paper notes
                              </button>
                            </div>
                          ) : null}

                          {needsManual ? (
                            <div className="manual-panel">
                              <div className="manual-panel-title">Manual resolution / upload</div>
                              <input
                                placeholder="Manual paper title"
                                value={manualForm.title || candidate.reference_title || candidate.paper_title || ''}
                                onChange={(event) => setManualPaperForm((current) => ({
                                  ...current,
                                  [candidate.candidate_id]: { ...current[candidate.candidate_id], title: event.target.value },
                                }))}
                              />
                              <input
                                placeholder="arXiv URL or ID, e.g. https://arxiv.org/abs/1611.09268"
                                value={manualForm.arxiv || ''}
                                onChange={(event) => setManualPaperForm((current) => ({
                                  ...current,
                                  [candidate.candidate_id]: { ...current[candidate.candidate_id], arxiv: event.target.value },
                                }))}
                              />
                              <input
                                placeholder="Year"
                                value={manualForm.year || ''}
                                onChange={(event) => setManualPaperForm((current) => ({
                                  ...current,
                                  [candidate.candidate_id]: { ...current[candidate.candidate_id], year: event.target.value },
                                }))}
                              />
                              <input
                                placeholder="Authors (comma separated)"
                                value={manualForm.authors || ''}
                                onChange={(event) => setManualPaperForm((current) => ({
                                  ...current,
                                  [candidate.candidate_id]: { ...current[candidate.candidate_id], authors: event.target.value },
                                }))}
                              />
                              <textarea
                                placeholder="Manual notes"
                                rows={2}
                                value={manualForm.notes || ''}
                                onChange={(event) => setManualPaperForm((current) => ({
                                  ...current,
                                  [candidate.candidate_id]: { ...current[candidate.candidate_id], notes: event.target.value },
                                }))}
                              />
                              <div className="manual-actions">
                                <button className="compact" onClick={() => linkArxiv(candidate)}>
                                  Link arXiv + Fetch
                                </button>
                                <button className="compact" onClick={() => createManualPaper(candidate)}>
                                  Create Manual Paper
                                </button>
                                <label className="upload-label">
                                  <Upload size={14} />
                                  {uploadingCandidateId === candidate.candidate_id ? 'Uploading...' : 'Upload PDF'}
                                  <input
                                    type="file"
                                    accept="application/pdf"
                                    onChange={(event) => uploadPdf(candidate, event.target.files?.[0])}
                                    hidden
                                  />
                                </label>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="glass-panel card section-block">
                  <div className="card-title">Benchmark Draft</div>
                  {!draft ? (
                    <div className="empty-substate">Run “Rebuild Draft” after extracting and resolving candidates.</div>
                  ) : (
                    <>
                      <div className="suggestions-box">
                        <div><strong>Suggested gold priors:</strong> {(draft.suggested_gold_prior_paper_ids || []).length}</div>
                        <div><strong>Suggested hard negatives:</strong> {(draft.suggested_hard_negative_ids || []).length}</div>
                        <div><strong>Suggested soft negatives:</strong> {(draft.suggested_soft_negative_ids || []).length}</div>
                      </div>

                      <div className="draft-field">
                        <label>Added-Information Label</label>
                        <select
                          value={draft.gold_added_information_label || ''}
                          onChange={(event) => {
                            applyDraftPatch({ gold_added_information_label: event.target.value });
                            saveDraftPatch({ gold_added_information_label: event.target.value, draft_status: 'needs_annotation' });
                          }}
                        >
                          <option value="">Unlabeled</option>
                          <option value="repackaging">repackaging</option>
                          <option value="incremental">incremental</option>
                          <option value="substantial">substantial</option>
                        </select>
                      </div>

                      <div className="draft-field">
                        <label>Gold Prior Support ACUs</label>
                        <textarea
                          rows={6}
                          value={(draft.gold_prior_support_acus || []).join('\n')}
                          onChange={(event) => applyDraftPatch({ gold_prior_support_acus: event.target.value.split('\n').filter(Boolean) })}
                          onBlur={(event) => saveDraftPatch({ gold_prior_support_acus: event.target.value.split('\n').filter(Boolean) })}
                        />
                      </div>

                      <div className="draft-field">
                        <label>Gold Prior Dataset Names</label>
                        <textarea
                          rows={3}
                          value={(draft.gold_prior_dataset_names || []).join('\n')}
                          onChange={(event) => applyDraftPatch({ gold_prior_dataset_names: event.target.value.split('\n').filter(Boolean) })}
                          onBlur={(event) => saveDraftPatch({ gold_prior_dataset_names: event.target.value.split('\n').filter(Boolean) })}
                        />
                      </div>

                      <div className="draft-field">
                        <label>Annotation Notes</label>
                        <textarea
                          rows={5}
                          value={draft.annotation_notes || ''}
                          onChange={(event) => applyDraftPatch({ annotation_notes: event.target.value })}
                          onBlur={(event) => saveDraftPatch({ annotation_notes: event.target.value })}
                        />
                      </div>

                      <div className="draft-field">
                        <label>Linked Prior Papers</label>
                        <div className="linked-paper-list">
                          {(draft.linked_prior_papers || []).map((paper) => (
                            <div key={paper.paper_id} className="linked-paper-card">
                              <div className="linked-paper-title">{paper.title}</div>
                              <div className="linked-paper-meta">{paper.arxiv_id || paper.paper_id}</div>
                              <div className="linked-paper-datasets">{(paper.dataset_names || []).join(', ')}</div>
                              <ul className="acu-list compact">
                                {(paper.acus || []).slice(0, 4).map((acu) => <li key={acu}>{acu}</li>)}
                              </ul>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="draft-actions">
                        <button className="primary" disabled={saving} onClick={markComplete}>
                          <Save size={16} />
                          Save / Mark Complete
                        </button>
                      </div>
                    </>
                  )}
                </div>
              </section>

              <section className="glass-panel card section-block">
                <div className="card-title">Job Monitor</div>
                <div className="job-list">
                  {jobs.slice().reverse().slice(0, 10).map((job) => (
                    <div key={job.job_id} className="job-row">
                      <div>
                        <strong>{job.job_type}</strong>
                        <div className="job-row-meta">{job.query_paper_id || 'bulk'} · {job.job_id}</div>
                      </div>
                      <div className="job-row-status">
                        <JobBadge status={job.status} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section className="glass-panel card section-block">
                <div className="card-title">Job Output</div>
                <pre className="job-output">{jobOutput || 'No job run yet.'}</pre>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default BenchmarkBuilder;
