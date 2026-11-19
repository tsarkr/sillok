// Vercel Serverless Function: Proxy historical archive search request to live Sillok engine

export const config = {
  maxDuration: 30,
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  const backend = (process.env.SILLOK_BACKEND_URL || 'https://api-sillok.tsar.kr').replace(/\/$/, '');

  try {
    const isPost = req.method === 'POST';
    const qs = !isPost && req.url && req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
    const response = await fetch(`${backend}/api/search${qs}`, {
      method: isPost ? 'POST' : 'GET',
      headers: isPost ? { 'Content-Type': 'application/json' } : undefined,
      body: isPost ? (typeof req.body === 'string' ? req.body : JSON.stringify(req.body)) : undefined,
    });

    const text = await response.text();
    let data;
    try {
      data = JSON.parse(text);
    } catch {
      return res.status(503).json({
        error: `실록 백엔드 서버가 자동 복구(재기동) 중입니다 (HTTP ${response.status}). 약 30~40초 후 다시 검색해 주세요.`,
      });
    }
    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(502).json({
      error: `실록 백엔드 서버 연결 실패: ${error.message}. 잠시 후 다시 시도해 주세요.`,
    });
  }
}
