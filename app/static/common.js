// 首页和股票页共用：工具中文名、建 DOM 元素、结构化回答卡片、流式提问
const NAMES = { get_quote:'查最新价格', get_history:'查历史走势', find_similar_history:'找历史相似情形', compare_with_index:'和大盘对比', search_news:'搜索新闻', set_price_alert:'设价位提醒',
  set_move_alert:'设波动提醒', list_alerts:'查看提醒', delete_alert:'删除提醒', watch:'修改关注', watch_topic:'修改关注话题', remember_preference:'记住偏好' };

// 结构化回答显示成卡片：结论、依据（每个数字注明来自哪个工具）、风险、执行结果、信心
function card(a) {
  const box = el('div', 'answer card');
  box.append(el('p', 'concl', a.conclusion));
  if (a.evidence.length) {
    const t = el('table');
    a.evidence.forEach(x => { const tr = el('tr'); tr.append(el('td', null, x.label), el('td', 'v', x.value), el('td', 's', NAMES[x.source] || x.source)); t.append(tr); });
    box.append(el('h3', null, '依据'), t);
  }
  if (a.risks.length) { const ul = el('ul'); a.risks.forEach(r => ul.append(el('li', null, r))); box.append(el('h3', null, '风险'), ul); }
  if (a.actions.length) {
    const ul = el('ul');
    a.actions.forEach(x => ul.append(el('li', x.ok ? 'ok' : 'fail', `${x.ok ? '✓' : '✗ 没有做成'} ${x.action}：${x.detail}`)));
    box.append(el('h3', null, '执行结果'), ul);
  }
  const conf = el('p'); conf.append(el('span', `badge c-${a.confidence}`, `信心 ${a.confidence}`), document.createTextNode(' ' + a.confidence_reason));
  box.append(el('h3', null, '信心'), conf, el('p', 'disc', '仅供学习参考，不构成投资建议。'));
  return box;
}

function el(tag, cls, text) { const x = document.createElement(tag); if (cls) x.className = cls; if (text != null) x.textContent = text; return x; }

// 调 /api/ask，把服务器推来的每个事件（tool_call / tool_result / answer / done / error）交给 show
async function streamAsk(question, session, show) {
  const headers = { 'Content-Type': 'application/json' };
  let pw = ''; try { pw = localStorage.pw || ''; } catch {}
  if (pw) headers.Authorization = 'Bearer ' + pw;
  try {
    const res = await fetch('/api/ask', { method:'POST', headers, body: JSON.stringify({ question, session_id: session }) });
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream:true });
      let i; while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i); buf = buf.slice(i + 2);
        if (line.startsWith('data: ')) show(JSON.parse(line.slice(6)));
      }
    }
  } catch (err) { show({ type:'error', message: String(err) }); }
}
