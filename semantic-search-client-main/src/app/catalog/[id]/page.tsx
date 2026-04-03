"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";

type ProductPagePayload = any;

export default function ProductPage({ params }: { params: Promise<{ id: string }> }) {
  const router = useRouter();
  const [productId, setProductId] = useState<number | null>(null);
  const CATALOG_API_BASE = process.env.NEXT_PUBLIC_CATALOG_URL || "http://localhost:8004/api/product_catalog";

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [product, setProduct] = useState<ProductPagePayload | null>(null);

  // Auto dark mode based on system preference
  useEffect(() => {
    const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", isDark);
  }, []);

  // Unwrap params Promise (Next.js 15+)
  useEffect(() => {
    params.then((resolvedParams) => {
      const id = Number(resolvedParams.id);
      setProductId(id);
    });
  }, [params]);

  useEffect(() => {
    if (productId === null || isNaN(productId)) return;

    async function fetchProduct() {
      setLoading(true);
      setError(null);
      try {
        const resp = await fetch(`${CATALOG_API_BASE}/product_page/${productId}`);
        if (!resp.ok) {
          const text = await resp.text();
          throw new Error(`Catalog error: ${resp.status} ${text}`);
        }
        const data = await resp.json();
        setProduct(data);
      } catch (err: any) {
        setError(err?.message || "Failed to load product");
      } finally {
        setLoading(false);
      }
    }
    fetchProduct();
  }, [productId, CATALOG_API_BASE]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-red-50 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <svg className="animate-spin h-16 w-16 text-red-600 mx-auto mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
          </svg>
          <p className="text-gray-600 dark:text-gray-400">Loading product details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-red-50 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 text-center">
          <svg className="w-16 h-16 text-red-600 dark:text-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Error Loading Product</h2>
          <p className="text-red-600 dark:text-red-400 mb-6">{error}</p>
          <button 
            onClick={() => router.back()}
            className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors font-medium"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  if (!product) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-red-50 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center p-6">
        <div className="max-w-md w-full bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 text-center">
          <svg className="w-16 h-16 text-gray-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Product Not Found</h2>
          <p className="text-gray-600 dark:text-gray-400 mb-6">The product you're looking for doesn't exist.</p>
          <button 
            onClick={() => router.back()}
            className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors font-medium"
          >
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-red-50 dark:from-gray-900 dark:to-gray-800 text-gray-900 dark:text-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center gap-4 mb-8">
          <button 
            onClick={() => router.back()}
            className="p-2 hover:bg-white dark:hover:bg-gray-800 rounded-lg transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </button>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-red-600 to-red-800 dark:from-red-400 dark:to-red-600 bg-clip-text text-transparent">
            Product Details
          </h1>
        </div>

        {/* Product Content */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden">
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 p-8">
            {/* Product Images */}
            <div className="lg:col-span-2 space-y-4">
              <div className="relative bg-gray-100 dark:bg-gray-700 rounded-xl overflow-hidden aspect-square">
                {product.primary_image ? (
                  <Image 
                    src={product.primary_image} 
                    alt={product.title || "product"} 
                    fill
                    className="object-contain p-8"
                    sizes="(max-width: 1024px) 100vw, 40vw"
                    priority
                  />
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <svg className="w-24 h-24 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                )}
              </div>

              {/* Additional Images */}
              {product.images && product.images.length > 1 && (
                <div className="grid grid-cols-4 gap-2">
                  {product.images.slice(0, 4).map((img: any, idx: number) => {
                    const imgUrl = img?.large || img?.hi_res || img?.thumb || "";
                    if (!imgUrl) return null;
                    return (
                      <div key={idx} className="relative aspect-square bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
                        <Image 
                          src={imgUrl} 
                          alt={`${product.title} - view ${idx + 1}`}
                          fill
                          className="object-contain p-2"
                          sizes="120px"
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Product Information */}
            <div className="lg:col-span-3 space-y-6">
              {/* Title & Category */}
              <div>
                <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                  {product.title}
                </h2>
                {product.main_category && (
                  <p className="text-gray-600 dark:text-gray-400">
                    Category: {product.main_category}
                  </p>
                )}
                {product.store_name && (
                  <p className="text-sm text-gray-500 dark:text-gray-500">
                    Sold by: {product.store_name}
                  </p>
                )}
              </div>

              {/* Price & Rating */}
              <div className="flex items-center gap-6 pb-6 border-b border-gray-200 dark:border-gray-700">
                <div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Price</div>
                  <div className="text-4xl font-bold text-red-600 dark:text-red-400">
                    {product.price ? `₹${product.price}` : "Price unavailable"}
                  </div>
                </div>
                
                {product.average_rating && (
                  <div>
                    <div className="text-sm text-gray-600 dark:text-gray-400 mb-1">Rating</div>
                    <div className="flex items-center gap-2">
                      <div className="flex">
                        {[...Array(5)].map((_, i) => (
                          <svg 
                            key={i} 
                            className={`w-6 h-6 ${i < Math.floor(product.average_rating) ? 'text-yellow-400 fill-current' : 'text-gray-300 dark:text-gray-600'}`}
                            viewBox="0 0 20 20"
                          >
                            <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                          </svg>
                        ))}
                      </div>
                      <span className="text-lg font-semibold text-gray-700 dark:text-gray-300">
                        {product.average_rating}
                      </span>
                      <span className="text-sm text-gray-500 dark:text-gray-400">
                        ({product.rating_number ?? 0} reviews)
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Attributes */}
              {product.archetypes && Object.keys(product.archetypes).length > 0 && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Product Attributes</h3>
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(product.archetypes).map(([key, values]) => (
                      <div key={key} className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4">
                        <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 capitalize">
                          {key.replace(/_/g, ' ')}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {(values as string[]).map((val, idx) => (
                            <span key={idx} className="inline-block bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 text-xs px-3 py-1 rounded-full">
                              {val}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Product Details */}
              {product.details && Object.keys(product.details).length > 0 && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Specifications</h3>
                  <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6 space-y-3">
                    {Object.entries(product.details).map(([key, value]) => (
                      <div key={key} className="flex border-b border-gray-200 dark:border-gray-600 last:border-0 pb-3 last:pb-0">
                        <span className="font-medium text-gray-700 dark:text-gray-300 min-w-[200px] capitalize">
                          {key.replace(/_/g, ' ')}:
                        </span>
                        <span className="text-gray-600 dark:text-gray-400 flex-1">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Videos */}
              {product.videos && product.videos.length > 0 && (
                <div>
                  <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">Product Videos</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {product.videos.map((video: any, idx: number) => (
                      <div key={idx} className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4">
                        <p className="text-sm text-gray-600 dark:text-gray-400">Video {idx + 1}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 pt-6">
                <a 
                  href={product.url ?? "#"} 
                  target="_blank" 
                  rel="noreferrer" 
                  className="flex-1 px-8 py-4 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white rounded-xl transition-all font-semibold text-lg text-center shadow-lg hover:shadow-xl"
                >
                  Buy Now on Store
                </a>
                <button 
                  onClick={() => router.back()}
                  className="px-8 py-4 border-2 border-red-600 text-red-600 dark:text-red-400 dark:border-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-xl transition-colors font-semibold text-lg"
                >
                  Back to Search
                </button>
              </div>

              {/* ASIN */}
              {product.parent_asin && (
                <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    Product ID: {product.parent_asin}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}