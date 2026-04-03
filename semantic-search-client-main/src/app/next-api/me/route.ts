import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest) {
  const username = req.cookies.get('username')?.value || '';
  return NextResponse.json({ username });
}
