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

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(502).json({
      error: `Failed to connect to Sillok archive search engine: ${error.message}`,
      hint: 'Ensure the Sillok backend engine is online.',
    });
  }
}
