const $ = s => document.getElementById(s);
let sid = null, busy = false, stream = true, ready = false, controller = null;
let sidebarOpen = false;

function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  sidebarOpen = !sidebarOpen;
  sidebar.classList.toggle('open', sidebarOpen);
  overlay?.classList.toggle('show', sidebarOpen);
}

function closeSidebar() {
  const sidebar = document.querySelector('.sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  sidebarOpen = false;
  sidebar.classList.remove('open');
  overlay?.classList.remove('show');
}

// ─── Init ───
(async function init(){
  await poll();
  setInterval(poll, 8000);
  setupInput();
  setupUpload();
  $('menuToggle')?.addEventListener('click', toggleSidebar);
  $('sidebarOverlay')?.addEventListener('click', closeSidebar);
})();

async function poll(){
  try{
    const r = await fetch('/api/status');
    if(!r.ok) return;
    const d = await r.json();
    const dot = $('eDot'), txt = $('eText');
    if(d.initialized){ dot.className='dot ready'; txt.textContent='引擎就绪'; ready=true; $('sendBtn').disabled=false; }
    else if(d.loading){ dot.className='dot loading'; txt.textContent='加载模型中...'; }
    else { dot.className='dot error'; txt.textContent='引擎未就绪'; }
    $('dCount').textContent = `${d.doc_count||0} 个文档片段`;
    if(d.llm_info?.name){ $('modelPill').textContent=d.llm_info.name; $('modelPill').style.display=''; }
    renderLaws(d.law_names||[]);
  }catch(e){ $('eDot').className='dot error'; $('eText').textContent='连接失败'; }
}

function renderLaws(names){
  const el=$('lawsList'), lb=$('lawsLabel');
  lb.textContent = names.length ? `已加载法律 (${names.length})` : '已加载法律';
  if(!names.length){ el.innerHTML='<div class="laws-empty">暂未加载文档</div>'; return; }
  el.innerHTML = names.map(n=>`<div class="law-item"><span class="law-dot"></span><span class="law-name" title="${esc(n)}">${esc(n)}</span><span class="law-del" onclick="delLaw('${esc(n)}',event)">✕</span></div>`).join('');
}

// ─── Upload ───
function setupUpload(){
  const z=$('uz'), f=$('fi');
  f.addEventListener('change',e=>handleFiles(e.target.files));
  z.addEventListener('dragover',e=>{e.preventDefault();z.classList.add('drag-over')});
  z.addEventListener('dragleave',()=>z.classList.remove('drag-over'));
  z.addEventListener('drop',e=>{e.preventDefault();z.classList.remove('drag-over');handleFiles(e.dataTransfer.files)});
}

async function handleFiles(files){
  const arr=[...files]; if(!arr.length) return;
  const bar=$('ubar'), fill=$('ubfill'), txt=document.querySelector('.upload-text'), orig=txt.textContent;
  bar.style.display='block';
  for(let i=0;i<arr.length;i++){
    fill.style.width=Math.round(i/arr.length*100)+'%';
    txt.textContent=arr.length>1?`上传 ${i+1}/${arr.length}`:'上传中...';
    await uploadOne(arr[i]);
  }
  fill.style.width='100%'; txt.textContent='上传完成';
  await poll();
  setTimeout(()=>{bar.style.display='none';fill.style.width='0%';txt.textContent=orig;$('fi').value=''},1200);
}

async function uploadOne(file){
  const fd=new FormData(); fd.append('file',file);
  try{
    const r=await fetch('/api/upload',{method:'POST',body:fd});
    if(!r.ok){const e=await r.json().catch(()=>({}));toast(`${file.name}：${e.detail||'上传失败'}`,'err');return}
    const d=await r.json();
    toast(`《${d.law_names?.join('》《')||file.name}》已加载 (${d.chunks_added} 片段)`,'ok');
  }catch(e){toast(`${file.name}：网络错误`,'err')}
}

async function delLaw(name,ev){
  ev.stopPropagation();
  if(!confirm(`确认删除《${name}》？`)) return;
  try{
    const r=await fetch(`/api/laws/${encodeURIComponent(name)}`,{method:'DELETE'});
    if(r.ok){toast('已删除','info');await poll()} else toast('删除失败','err');
  }catch(e){toast('删除失败','err')}
}

// ─── Input ───
function setupInput(){
  const b=$('ib'), c=$('cc');
  b.addEventListener('input',()=>{
    b.style.height='auto';b.style.height=Math.min(b.scrollHeight,140)+'px';
    const n=b.value.length;c.textContent=`${n} / 1000`;c.style.opacity=n?'1':'0';c.style.color=n>900?'var(--red)':'var(--fg3)';
  });
  b.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send()}});
}

function toggleStream(){stream=!stream;$('sState').textContent=stream?'开':'关';$('sState').style.color=stream?'var(--accent)':'var(--fg3)'}

function cancelRequest() {
  if (controller) {
    controller.abort();
    controller = null;
    toast('已取消', 'info');
    const think = document.querySelector('.msg.ai .thinking');
    if (think) think.parentElement.remove();
    busy = false;
    $('sendBtn').disabled = false;
    $('cancelBtn').style.display = 'none';
  }
}

// ─── Chat ───
function ask(t){$('ib').value=t;$('ib').dispatchEvent(new Event('input'));send()}
function newChat(){sid=null;$('sessPill').textContent='新对话';$('msgs').innerHTML='';$('msgs').style.display='none';$('welcome').style.display=''}

async function send(){
  const b=$('ib'), q=b.value.trim();
  if(!q||busy) return;
  if(!ready){toast('引擎尚未就绪','info');return}
  $('welcome').style.display='none';$('msgs').style.display='block';
  b.value='';b.style.height='auto';$('cc').style.opacity='0';
  addMsg('user',q);
  busy=true;$('sendBtn').disabled=true;
  $('cancelBtn').style.display='inline';
  controller = new AbortController();
  const tid=addThink();
  try{ stream ? await streamQ(q,tid) : await normalQ(q,tid) }
  catch(e){rmThink(tid);addMsg('ai','请求失败，请检查网络连接。');toast('请求失败','err')}
  finally{busy=false;$('sendBtn').disabled=!ready;$('cancelBtn').style.display='none';scroll()}
}

async function normalQ(q,tid){
  const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,session_id:sid}),signal:controller.signal});
  rmThink(tid);
  if(!r.ok){const e=await r.json().catch(()=>({}));addMsg('ai',`错误：${e.detail||r.statusText}`);return}
  const d=await r.json();sid=d.session_id;$('sessPill').textContent=`会话 ${sid?.slice(-6)||''}`;
  addMsg('ai',d.answer,d.sources);
}

async function streamQ(q,tid){
  const r=await fetch('/api/query/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,session_id:sid}),signal:controller.signal});
  if(!r.ok){rmThink(tid);addMsg('ai','请求失败');return}
  rmThink(tid);
  const el=addMsg('ai','',null,true), bub=el.querySelector('.bubble');
  const cur=document.createElement('span');cur.className='cursor';
  let text='',src=null;
  const reader=r.body.getReader(),dec=new TextDecoder();let buf='';
  while(true){
    const{done,value}=await reader.read();if(done)break;
    buf+=dec.decode(value,{stream:true});
    const lines=buf.split('\n');buf=lines.pop();
    for(const line of lines){
      if(!line.startsWith('data: '))continue;
      const p=line.slice(6);
      if(p==='[DONE]')continue;
      if(p.startsWith('[METADATA]')){
        try{const m=JSON.parse(p.slice(10));if(m.session_id){sid=m.session_id;$('sessPill').textContent=`会话 ${sid.slice(-6)}`}if(m.sources)src=m.sources}catch(e){}
        continue;
      }
      text+=p;bub.innerHTML=fmt(text);bub.appendChild(cur);scroll();
    }
  }
  cur.remove();bub.innerHTML=fmt(text);
  if(src?.length){el.querySelector('.msg-body').appendChild(buildSrc(src))}
  const ts=document.createElement('div');ts.className='msg-time';ts.textContent=now();el.querySelector('.msg-body').appendChild(ts);
}

// ─── DOM ───
function addMsg(role,text,src=null,streaming=false){
  const m=$('msgs'),d=document.createElement('div');d.className=`msg ${role}`;
  const av=document.createElement('div');av.className='avatar';av.textContent=role==='user'?'我':'律';
  const body=document.createElement('div');body.className='msg-body';
  const bub=document.createElement('div');bub.className='bubble';
  if(!streaming) bub.innerHTML=role==='ai'?fmt(text):esc(text).replace(/\n/g,'<br>');
  body.appendChild(bub);
  if(!streaming){
    if(src?.length) body.appendChild(buildSrc(src));
    const ts=document.createElement('div');ts.className='msg-time';ts.textContent=now();body.appendChild(ts);
  }
  d.appendChild(av);d.appendChild(body);m.appendChild(d);

  if (role === 'user') {
    const actions = document.createElement('div');
    actions.className = 'msg-actions';
    actions.innerHTML = `
      <button class="msg-action" onclick="copyMessage('${esc(text)}')" title="复制">📋</button>
      <button class="msg-action" onclick="editMessage(this)" title="编辑">✏️</button>
    `;
    body.appendChild(actions);
    
    const editDiv = document.createElement('div');
    editDiv.className = 'msg-edit';
    editDiv.innerHTML = `
      <textarea>${esc(text)}</textarea>
      <div class="msg-edit-btns">
        <button class="confirm-edit" onclick="confirmEdit(this)">确认</button>
        <button class="cancel-edit" onclick="cancelEdit(this)">取消</button>
      </div>
    `;
    body.appendChild(editDiv);
  }

  scroll();return d;
}

function copyMessage(text) {
  navigator.clipboard.writeText(text).then(() => {
    toast('已复制到剪贴板', 'ok');
  });
}

function editMessage(btn) {
  const msg = btn.closest('.msg');
  msg.classList.add('editing');
}

function confirmEdit(btn) {
  const msg = btn.closest('.msg');
  const newText = msg.querySelector('textarea').value.trim();
  if (newText) {
    msg.classList.remove('editing');
    msg.querySelector('.bubble').innerHTML = newText.replace(/\n/g, '<br>');
    const q = newText;
    msg.remove();
    sendEdit(q);
  }
}

function cancelEdit(btn) {
  const msg = btn.closest('.msg');
  msg.classList.remove('editing');
}

async function sendEdit(q) {
  const b = $('ib');
  b.value = q;
  $('welcome').style.display = 'none';
  $('msgs').style.display = 'block';
  addMsg('user', q);
  busy = true;
  $('sendBtn').disabled = true;
  const tid = addThink();
  try {
    stream ? await streamQ(q, tid) : await normalQ(q, tid);
  } catch (e) {
    rmThink(tid);
    addMsg('ai', '请求失败，请检查网络连接。');
    toast('请求失败', 'err');
  } finally {
    busy = false;
    $('sendBtn').disabled = !ready;
    scroll();
  }
}

function addThink(){
  const m=$('msgs'),id='t'+Date.now(),d=document.createElement('div');
  d.className='msg ai';d.id=id;
  d.innerHTML=`<div class="avatar">律</div><div class="msg-body"><div class="thinking"><div class="thinking-dots"><div class="tdot"></div><div class="tdot"></div><div class="tdot"></div></div><span class="thinking-text">正在检索法律条文...</span></div></div>`;
  m.appendChild(d);scroll();return id;
}
function rmThink(id){document.getElementById(id)?.remove()}

function buildSrc(sources){
  const p=document.createElement('div');p.className='sources';
  const t=document.createElement('div');t.className='sources-toggle';
  t.innerHTML=`<span class="sources-arrow">▶</span> 引用条文 (${sources.length})`;
  t.onclick=function(){this.classList.toggle('open');p.querySelector('.sources-body').classList.toggle('open')};
  const b=document.createElement('div');b.className='sources-body';
  sources.forEach(s=>{
    const i=document.createElement('div');i.className='src-item';
    i.innerHTML=`<div class="src-head"><span class="src-law">${esc(s.law_name)}</span><span class="src-article">${esc(s.article)}</span></div><div class="src-content">${esc(s.content)}</div>`;
    b.appendChild(i);
  });
  p.appendChild(t);p.appendChild(b);return p;
}

function fmt(t){
  if(!t)return'';
  t = t.replace(/```([\s\S]*?)```/g, (match, code) => {
    return '<pre><code class="language-text">' + code.trim() + '</code></pre>';
  });
  t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
  t = t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/【法律分析】/g,'<div class="sh">【法律分析】</div>')
    .replace(/【适用条文】/g,'<div class="sh">【适用条文】</div>')
    .replace(/【法律结论】/g,'<div class="sh_alt">【法律结论】</div>')
    .replace(/【实务建议】/g,'<div class="sh_alt">【实务建议】</div>')
    .replace(/【([^】]+)】/g,'<div class="sh">【$1】</div>')
    .replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
    .replace(/^[-•]\s+(.+)$/gm,'<div style="padding-left:10px;margin:2px 0">· $1</div>')
    .replace(/\n/g,'<br>');
  setTimeout(() => {
    document.querySelectorAll('.msg.ai pre code').forEach(block => {
      hljs.highlightElement(block);
    });
  }, 0);
  return t;
}

function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;')}
function scroll(){$('chatArea').scrollTop=$('chatArea').scrollTop=$('chatArea').scrollHeight}
function now(){return new Date().toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit'})}

// ─── History ───
function showHistory(){$('mainPanel').style.display='none';$('histPanel').classList.add('show');loadHist()}
function hideHistory(){$('histPanel').classList.remove('show');$('mainPanel').style.display=''}

async function loadHist(){
  const el=$('histList');el.innerHTML='<div class="history-empty">加载中...</div>';
  try{
    const r=await fetch('/api/history'),d=await r.json();
    if(!d.sessions?.length){el.innerHTML='<div class="history-empty">暂无记录</div>';return}
    el.innerHTML='';
    d.sessions.forEach(s=>{
      const i=document.createElement('div');i.className='history-item'+(s.id===sid?' active':'');
      const title=s.title||`会话 ${s.id.slice(-6)}`;
      i.innerHTML=`<div class="hi-title">${esc(title)}</div><div class="hi-preview">${esc(s.last_message||'')}</div><div class="hi-meta"><span class="hi-count">${s.message_count||0} 条消息</span></div><div class="hi-actions"><span class="hi-action ren" onclick="renSession('${s.id}','${esc(title)}',event)">✎</span><span class="hi-action del" onclick="delSession('${s.id}',event)">✕</span></div>`;
      i.onclick=()=>loadSession(s.id);el.appendChild(i);
    });
  }catch(e){el.innerHTML='<div class="history-empty">加载失败</div>';toast('获取历史失败','err')}
}

async function loadSession(id){
  try{
    const r=await fetch(`/api/history/${id}`);if(!r.ok){toast('加载失败','err');return}
    const d=await r.json();
    sid=id;$('sessPill').textContent=`会话 ${id.slice(-6)}`;
    const m=$('msgs');m.innerHTML='';
    $('welcome').style.display='none';m.style.display='block';
    for(const msg of d.messages){
      const role=msg.type==='human'?'user':'ai';
      const content=msg.data?.content||'';
      addMsg(role,content);
    }
    hideHistory();toast('已恢复对话','ok');
  }catch(e){toast('加载失败','err')}
}

async function delSession(id,ev){
  ev.stopPropagation();if(!confirm('确认删除此会话？'))return;
  try{const r=await fetch(`/api/history/${id}`,{method:'DELETE'});
    if(r.ok){toast('已删除','info');await loadHist();if(sid===id)newChat()}else toast('删除失败','err');
  }catch(e){toast('删除失败','err')}
}

async function renSession(id,cur,ev){
  ev.stopPropagation();const t=prompt('输入新名称:',cur);if(!t||t===cur)return;
  try{const r=await fetch(`/api/history/${id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({title:t})});
    if(r.ok){toast('已重命名','ok');await loadHist()}else toast('重命名失败','err');
  }catch(e){toast('重命名失败','err')}
}

async function clearAll(){
  if(!confirm('确认清空所有历史记录？'))return;
  try{const r=await fetch('/api/history',{method:'DELETE'});
    if(r.ok){toast('已清空','info');await loadHist();newChat()}else toast('操作失败','err');
  }catch(e){toast('操作失败','err')}
}

// ─── Toast ───
function toast(msg,type='info'){
  const c=$('toasts'),el=document.createElement('div');
  el.className=`toast ${type}`;
  const icons={ok:'✓',err:'✕',info:'ℹ'};
  el.innerHTML=`<span>${icons[type]}</span><span>${esc(msg)}</span>`;
  c.appendChild(el);
  setTimeout(()=>{el.style.animation='slideIn .25s ease reverse both';setTimeout(()=>el.remove(),240)},3000);
}