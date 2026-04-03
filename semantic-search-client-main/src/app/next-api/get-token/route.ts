import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const authToken = req.cookies.get('authToken')?.value || '';
  return NextResponse.json({ authToken });
}