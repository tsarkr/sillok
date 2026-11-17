// Vercel Serverless Function: Proxy generation request to live Sillok engine

export const config = {
  maxDuration: 60,
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const backend = (process.env.SILLOK_BACKEND_URL || 'https://reg-township-clan-founded.trycloudflare.com').replace(/\/$/, '');

  try {
    const response = await fetch(`${backend}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: typeof req.body === 'string' ? req.body : JSON.stringify(req.body),
    });

    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    return res.status(502).json({
      error: `Failed to connect to Sillok inference engine at ${backend}: ${error.message}`,
      backend,
      hint: 'Ensure your Sillok local/Colab server or Cloudflare tunnel is running.',
    });
  }
}
