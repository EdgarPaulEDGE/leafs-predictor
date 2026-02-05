// EDGE NHL Service Worker v3 - FORCE UPDATE
var CACHE_NAME = 'edge-nhl-v3';

// Delete ALL old caches on activate
self.addEventListener('activate', function(e) {
  e.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(
        keys.map(function(k) {
          return caches.delete(k);
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('install', function(e) {
  self.skipWaiting();
});

// Network-first for everything
self.addEventListener('fetch', function(e) {
  e.respondWith(
    fetch(e.request).catch(function() {
      return caches.match(e.request);
    })
  );
});
