import React, { useState } from 'react';

export default function App(){
  const [q, setQ] = useState('');
  const [res, setRes] = useState(null);
  const [loading, setLoading] = useState(false);

  async function sendQuery(){
    setLoading(true);
    setRes(null);
    try{
      const r = await fetch('/api/query', {
        method: 'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify({question: q})
      });
      const j = await r.json();
      setRes(j);
    }catch(e){
      setRes({error: String(e)});
    }finally{
      setLoading(false);
    }
  }

  return (
    <div style={{fontFamily:'Inter, sans-serif', padding:24}}>
      <h1>Adaptive RAG — Demo</h1>
      <p>Type a question and see controller decisions + retrieval trace.</p>
      <textarea value={q} onChange={e=>setQ(e.target.value)} rows={4} style={{width:'100%'}} />
      <div style={{marginTop:12}}>
        <button onClick={sendQuery} disabled={loading || !q}>Run query</button>
      </div>
      <pre style={{marginTop:12, whiteSpace:'pre-wrap'}}>{JSON.stringify(res, null, 2)}</pre>
    </div>
  );
}
