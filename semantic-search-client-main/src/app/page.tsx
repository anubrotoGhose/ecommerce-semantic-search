"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import Image from "next/image";

type SearchRequest = {
  user_query: string;
  page?: number;
  page_size?: number;
};

type Product = {
  id: number;
  title?: string;
  price?: number | string;
  average_rating?: number;
  rating_number?: number;
  image?: string;
  primary_image?: string;
  store_name?: string;
  url?: string;
};

type SearchResponse = {
  products: Product[];
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
};

export default function HomePage() {
  const PUBLIC_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://10.169.101.65:8000";
  const CATALOG_API_BASE = process.env.NEXT_PUBLIC_CATALOG_URL || "http://localhost:8004/api/product_catalog";

  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  const [modalOpen, setModalOpen] = useState(false);
  const [selectedProductId, setSelectedProductId] = useState<number | null>(null);
  const [selectedProductDetails, setSelectedProductDetails] = useState<any | null>(null);
  const [modalLoading, setModalLoading] = useState(false);
  const [modalError, setModalError] = useState<string | null>(null);

  // Auto dark mode based on system preference
  useEffect(() => {
    const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.classList.toggle("dark", isDark);
  }, []);


async function handleSearch(e?: React.FormEvent, page: number = 1) {
  if (e) e.preventDefault();
  setError(null);
  const user_query = query.trim();
  if (!user_query) {
    setError("Please enter a search query.");
    return;
  }

  const payload: SearchRequest = { 
    user_query,
    page,
    page_size: pageSize
  };

  try {
    setLoading(true);
    const resp = await fetch(`${PUBLIC_API_URL}/search/public`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`Server error: ${resp.status} ${text}`);
    }

    const data = await resp.json();
    
    // Add validation here
    console.log("API Response:", data); // Debug log
    
    setSearchResponse(data);
    setCurrentPage(page);
    
    // Scroll to top of results
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err: any) {
    setError(err?.message || "Unknown error");
    setSearchResponse(null);
  } finally {
    setLoading(false);
  }
}


  function handlePageChange(newPage: number) {
    handleSearch(undefined, newPage);
  }

  function openModal(productId: number) {
    setSelectedProductId(productId);
    setModalOpen(true);
    fetchProductDetails(productId);
  }

  async function fetchProductDetails(productId: number) {
    setSelectedProductDetails(null);
    setModalError(null);
    setModalLoading(true);
    try {
      const resp = await fetch(`${CATALOG_API_BASE}/product_page/${productId}`);
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Catalog error: ${resp.status} ${text}`);
      }
      const data = await resp.json();
      setSelectedProductDetails(data);
    } catch (err: any) {
      setModalError(err?.message || "Failed to fetch product details");
    } finally {
      setModalLoading(false);
    }
  }

  // Generate page numbers to display
  function getPageNumbers() {
    if (!searchResponse) return [];
    
    const { page, total_pages } = searchResponse;
    const pages: (number | string)[] = [];
    
    if (total_pages <= 7) {
      // Show all pages if 7 or fewer
      for (let i = 1; i <= total_pages; i++) {
        pages.push(i);
      }
    } else {
      // Always show first page
      pages.push(1);
      
      if (page > 3) {
        pages.push('...');
      }
      
      // Show pages around current page
      for (let i = Math.max(2, page - 1); i <= Math.min(total_pages - 1, page + 1); i++) {
        pages.push(i);
      }
      
      if (page < total_pages - 2) {
        pages.push('...');
      }
      
      // Always show last page
      pages.push(total_pages);
    }
    
    return pages;
  }

  const products = searchResponse?.products || null;

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-red-50 dark:from-gray-900 dark:to-gray-800 text-gray-900 dark:text-gray-100">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <header className="mb-12 text-center">
          <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-red-600 to-red-800 dark:from-red-400 dark:to-red-600 bg-clip-text text-transparent mb-3">
            Product Discovery
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Find exactly what you're looking for with intelligent search
          </p>
        </header>

        {/* Search Form */}
        <form onSubmit={(e) => handleSearch(e, 1)} className="mb-10 max-w-4xl mx-auto">
          <div className="relative">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for products... (e.g., waterproof running shoes)"
              className="w-full rounded-xl border-2 border-gray-200 dark:border-gray-700 px-6 py-4 bg-white dark:bg-gray-800 shadow-lg focus:outline-none focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all text-lg"
            />
            <button
              type="submit"
              disabled={loading}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-8 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white rounded-lg shadow-md hover:shadow-lg disabled:opacity-60 disabled:cursor-not-allowed transition-all font-medium"
            >
              {loading ? (
                <div className="flex items-center gap-2">
                  <svg className="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  <span>Searching</span>
                </div>
              ) : (
                "Search"
              )}
            </button>
          </div>

          {/* Items per page selector */}
          {searchResponse && (
            <div className="flex items-center justify-end gap-2 mt-4 text-sm">
              <label className="text-gray-600 dark:text-gray-400">Items per page:</label>
              <select
                value={pageSize}
                onChange={(e) => {
                  const newPageSize = Number(e.target.value);
                  setPageSize(newPageSize);
                  
                  // Use the new value directly, not from state
                  const payload: SearchRequest = { 
                    user_query: query.trim(),
                    page: 1,
                    page_size: newPageSize // Use the new value directly
                  };
                  
                  // Make the API call with the new value
                  fetch(`${PUBLIC_API_URL}/search`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                  })
                    .then(resp => resp.json())
                    .then(data => {
                      setSearchResponse(data);
                      setCurrentPage(1);
                    })
                    .catch(err => {
                      setError(err?.message || "Unknown error");
                    });
                }}
                className="px-3 py-1 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
              >

                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
              </select>
            </div>
          )}
        </form>

        {/* Error Message */}
        {error && (
          <div className="max-w-4xl mx-auto mb-6 bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 text-red-800 dark:text-red-200 p-4 rounded-lg shadow">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              {error}
            </div>
          </div>
        )}

        {/* Results Info */}
        {searchResponse && (
          <div className="mb-6 flex items-center justify-between">
            <div className="text-gray-600 dark:text-gray-400">
              Showing {((searchResponse.page - 1) * searchResponse.page_size) + 1} - {Math.min(searchResponse.page * searchResponse.page_size, searchResponse.total_count)} of {Number(searchResponse.total_count || 0).toLocaleString()} products
            </div>
            <div className="text-sm text-gray-500 dark:text-gray-500">
              Page {searchResponse.page} of {searchResponse.total_pages}
            </div>
          </div>
        )}

        {/* Results Section */}
        <section>
          {products && products.length > 0 ? (
            <div>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {products.map((p) => (
                  <article 
                    key={p.id} 
                    className="group bg-white dark:bg-gray-800 rounded-xl shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-100 dark:border-gray-700 hover:border-red-200 dark:hover:border-red-800"
                  >
                    {/* Product Image */}
                    <div className="relative h-64 bg-gray-100 dark:bg-gray-700 overflow-hidden">
                      {p.image || p.primary_image ? (
                        <Image 
                          src={p.image || p.primary_image || ''} 
                          alt={p.title || "product image"} 
                          fill
                          className="object-contain p-4 group-hover:scale-105 transition-transform duration-300"
                          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 25vw"
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <svg className="w-16 h-16 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                        </div>
                      )}
                    </div>

                    {/* Product Info */}
                    <div className="p-4">
                      <h3 className="font-semibold text-gray-900 dark:text-white line-clamp-2 mb-3 min-h-[3rem]">
                        {p.title ?? "Untitled Product"}
                      </h3>

                      <div className="space-y-2 mb-4">
                        <div className="flex items-center justify-between">
                          <span className="text-2xl font-bold text-red-600 dark:text-red-400">
                            {p.price ? `₹${p.price}` : "Price unavailable"}
                          </span>
                          {p.average_rating && (
                            <div className="flex items-center gap-1 text-sm">
                              <svg className="w-5 h-5 text-yellow-400 fill-current" viewBox="0 0 20 20">
                                <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                              </svg>
                              <span className="font-medium text-gray-700 dark:text-gray-300">
                                {p.average_rating}
                              </span>
                              {p.rating_number && (
                                <span className="text-gray-500 dark:text-gray-400">
                                  ({p.rating_number})
                                </span>
                              )}
                            </div>
                          )}
                        </div>
                        
                        {p.store_name && (
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            by {p.store_name}
                          </div>
                        )}
                      </div>

                      {/* Action Buttons */}
                      <div className="flex gap-2">
                        <button
                          onClick={() => openModal(p.id)}
                          className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors font-medium text-sm"
                        >
                          Quick View
                        </button>
                        <Link 
                          href={`/catalog/${p.id}`}
                          className="px-4 py-2 border-2 border-red-600 text-red-600 dark:text-red-400 dark:border-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors font-medium text-sm"
                        >
                          Details
                        </Link>
                      </div>
                    </div>
                  </article>
                ))}
              </div>

              {/* Pagination Controls */}
              {searchResponse && searchResponse.total_pages > 1 && (
                <div className="mt-10 flex items-center justify-center gap-2">
                  {/* Previous Button */}
                  <button
                    onClick={() => handlePageChange(currentPage - 1)}
                    disabled={!searchResponse.has_previous || loading}
                    className="px-4 py-2 border-2 border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                  </button>

                  {/* Page Numbers */}
                  {getPageNumbers().map((pageNum, idx) => (
                    pageNum === '...' ? (
                      <span key={`ellipsis-${idx}`} className="px-4 py-2 text-gray-500">...</span>
                    ) : (
                      <button
                        key={pageNum}
                        onClick={() => handlePageChange(pageNum as number)}
                        disabled={loading}
                        className={`min-w-[44px] px-4 py-2 rounded-lg transition-colors font-medium ${
                          currentPage === pageNum
                            ? 'bg-red-600 text-white shadow-md'
                            : 'border-2 border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                        } disabled:opacity-60`}
                      >
                        {pageNum}
                      </button>
                    )
                  ))}

                  {/* Next Button */}
                  <button
                    onClick={() => handlePageChange(currentPage + 1)}
                    disabled={!searchResponse.has_next || loading}
                    className="px-4 py-2 border-2 border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </button>
                </div>
              )}

              {/* Jump to Page */}
              {searchResponse && searchResponse.total_pages > 10 && (
                <div className="mt-6 flex items-center justify-center gap-3">
                  <label className="text-sm text-gray-600 dark:text-gray-400">Jump to page:</label>
                  <input
                    type="number"
                    min={1}
                    max={searchResponse.total_pages}
                    placeholder="Page"
                    className="w-20 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-center"
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        const page = parseInt((e.target as HTMLInputElement).value);
                        if (page >= 1 && page <= searchResponse.total_pages) {
                          handlePageChange(page);
                          (e.target as HTMLInputElement).value = '';
                        }
                      }
                    }}
                  />
                  <span className="text-sm text-gray-500 dark:text-gray-500">
                    of {searchResponse.total_pages}
                  </span>
                </div>
              )}
            </div>
          ) : products && products.length === 0 ? (
            <div className="text-center py-16">
              <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-xl text-gray-600 dark:text-gray-400">No products found</p>
              <p className="text-gray-500 dark:text-gray-500 mt-2">Try adjusting your search terms</p>
            </div>
          ) : (
            <div className="text-center py-20">
              <svg className="mx-auto h-20 w-20 text-red-300 dark:text-red-700 mb-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <p className="text-xl text-gray-600 dark:text-gray-400 mb-2">Start your search</p>
              <p className="text-gray-500 dark:text-gray-500">Enter a query above to find products</p>
            </div>
          )}
        </section>
      </div>

      {/* Modal - unchanged */}
      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div 
            className="absolute inset-0" 
            onClick={() => setModalOpen(false)}
          />

          <div className="relative max-w-4xl w-full bg-white dark:bg-gray-800 rounded-2xl shadow-2xl overflow-hidden z-10 max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-6 flex items-start justify-between z-10">
              <div className="flex-1 pr-4">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-1">
                  {selectedProductDetails?.title ?? "Product Details"}
                </h2>
                {selectedProductDetails?.main_category && (
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {selectedProductDetails.main_category}
                  </p>
                )}
              </div>
              <button 
                onClick={() => setModalOpen(false)}
                className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6">
              {modalLoading ? (
                <div className="flex items-center justify-center py-16">
                  <svg className="animate-spin h-12 w-12 text-red-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                </div>
              ) : modalError ? (
                <div className="text-center py-16">
                  <div className="text-red-600 dark:text-red-400">{modalError}</div>
                </div>
              ) : selectedProductDetails ? (
                <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
                  <div className="md:col-span-2">
                    <div className="relative bg-gray-100 dark:bg-gray-700 rounded-xl overflow-hidden aspect-square">
                      {selectedProductDetails.primary_image ? (
                        <Image 
                          src={selectedProductDetails.primary_image} 
                          alt={selectedProductDetails.title || "product"} 
                          fill
                          className="object-contain p-6"
                          sizes="(max-width: 768px) 100vw, 40vw"
                        />
                      ) : (
                        <div className="flex items-center justify-center h-full">
                          <svg className="w-20 h-20 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                          </svg>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="md:col-span-3 space-y-4">
                    <div>
                      <div className="text-3xl font-bold text-red-600 dark:text-red-400 mb-2">
                        {selectedProductDetails.price ? `₹${selectedProductDetails.price}` : "Price unavailable"}
                      </div>
                      
                      {selectedProductDetails.average_rating && (
                        <div className="flex items-center gap-2">
                          <div className="flex">
                            {[...Array(5)].map((_, i) => (
                              <svg 
                                key={i} 
                                className={`w-5 h-5 ${i < Math.floor(selectedProductDetails.average_rating) ? 'text-yellow-400 fill-current' : 'text-gray-300 dark:text-gray-600'}`}
                                viewBox="0 0 20 20"
                              >
                                <path d="M10 15l-5.878 3.09 1.123-6.545L.489 6.91l6.572-.955L10 0l2.939 5.955 6.572.955-4.756 4.635 1.123 6.545z" />
                              </svg>
                            ))}
                          </div>
                          <span className="text-sm text-gray-600 dark:text-gray-400">
                            {selectedProductDetails.average_rating} ({selectedProductDetails.rating_number ?? 0} ratings)
                          </span>
                        </div>
                      )}
                    </div>

                    {selectedProductDetails.archetypes && Object.keys(selectedProductDetails.archetypes).length > 0 && (
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Attributes</h3>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(selectedProductDetails.archetypes).map(([k, vals]) => (
                            <div key={k} className="inline-flex items-center gap-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 px-3 py-1 rounded-full">
                              <span className="text-xs font-medium text-red-700 dark:text-red-300">{k}:</span>
                              <span className="text-xs text-red-600 dark:text-red-400">{(vals as string[]).join(", ")}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {selectedProductDetails.details && Object.keys(selectedProductDetails.details).length > 0 && (
                      <div>
                        <h3 className="font-semibold text-gray-900 dark:text-white mb-3">Product Details</h3>
                        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-4 space-y-2">
                          {Object.entries(selectedProductDetails.details).map(([key, value]) => (
                            <div key={key} className="flex gap-2">
                              <span className="font-medium text-gray-700 dark:text-gray-300 min-w-[120px]">{key}:</span>
                              <span className="text-gray-600 dark:text-gray-400">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="flex gap-3 pt-4">
                      <a 
                        href={selectedProductDetails.url ?? "#"} 
                        target="_blank" 
                        rel="noreferrer" 
                        className="flex-1 px-6 py-3 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 text-white rounded-lg transition-all font-medium text-center"
                      >
                        View on Store
                      </a>
                      <Link 
                        href={`/catalog/${selectedProductDetails.id}`}
                        className="px-6 py-3 border-2 border-red-600 text-red-600 dark:text-red-400 dark:border-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors font-medium"
                      >
                        Full Details
                      </Link>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}