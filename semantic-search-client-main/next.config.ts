import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
      {
        protocol: 'http',
        hostname: '**',
      },
    ],
    // Disable image optimization to bypass SSL certificate issues
    unoptimized: true,
  },
  // Suppress source map warnings
  // webpack: (config, { isServer }) => {
  //   if (!isServer) {
  //     config.devtool = false;
  //   }
  //   return config;
  // },
};

export default nextConfig;
