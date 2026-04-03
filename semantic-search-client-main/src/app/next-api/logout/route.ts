import { NextResponse } from 'next/server';

export async function POST() {
  const response = NextResponse.json({ success: true });
  response.cookies.set('authToken', '', {
    httpOnly: true,
    sameSite: 'strict',
    expires: new Date(0),
    path: '/',
  });
  response.cookies.set('username', '', {
    httpOnly: true,
    expires: new Date(0),
    sameSite: 'strict',
    path: '/',
  });
  return response;
}