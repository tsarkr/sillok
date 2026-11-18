// Vercel Serverless Function: Proxy status request to live Sillok engine

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  const backend = (process.env.SILLOK_BACKEND_URL || 'https://camp-standard-strengthening-safer.trycloudflare.com').replace(/\/$/, '');

  try {
    const response = await fetch(`${backend}/api/status`);
    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(502).json({
      status: 'offline',
      backend,
      error: error.message,
    });
  }
}
