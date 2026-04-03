import { NextRequest, NextResponse } from 'next/server';

// Sanitize input to prevent XSS
const sanitizeInput = (input: string) => input.replace(/[<>]/g, '');

export async function POST(req: NextRequest) {
  const { token, username } = await req.json();
  const response = NextResponse.json({ success: true });

  // Sanitize username
  const sanitizedUsername = sanitizeInput(username);

  // Set HTTP-only cookie for auth token
  response.cookies.set('authToken', token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/',
    maxAge: 60 * 60 * 24, // 1 day
  });

  // Set HTTP-only cookie for username (JS can't access, but safer)
  response.cookies.set('username', sanitizedUsername, {
    httpOnly: true, // Now HTTP-only!
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    path: '/',
    maxAge: 60 * 60 * 24, // 1 day
  });

  return response;
}
