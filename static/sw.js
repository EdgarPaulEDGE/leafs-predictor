// EDGE NHL Service Worker – Ultra-Speed Offline-Cache v2
var CACHE_NAME = 'edge-nhl-v2';

// Kritische Assets die SOFORT gecached werden
var PRECACHE = [
  '/static/style.min.css',
  '/static/edge-logo.svg',
  '/static/manifest.json'
];

// NHL CDN URLs die wir cachen (Team Logos, Headshots)
var NHL_CDN_CACHE = 'nhl-cdn-v1';

self.addEventListener('install', function(e) {
  e.waitUntil(
    caches.open(CACHE_NAME).then(function(cache) {
      return cache.addAll(PRECACHE);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', function(e) {
  e.waitUntil(
    caches.keys().then(function(keys) {
      return Promise.all(
        keys.filter(function(k) {
          return k !== CACHE_NAME && k !== NHL_CDN_CACHE;
        }).map(function(k) {
          return caches.delete(k);
        })
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', function(e) {
  var url = e.request.url;

  // NHL CDN Assets (Logos, Headshots): Cache-First (langlebig)
  if (url.includes('assets.nhle.com')) {
    e.respondWith(
      caches.open(NHL_CDN_CACHE).then(function(cache) {
        return cache.match(e.request).then(function(cached) {
          if (cached) return cached;
          return fetch(e.request).then(function(resp) {
            if (resp.status === 200) {
              cache.put(e.request, resp.clone());
            }
            return resp;
          }).catch(function() {
            // Fallback: Generisches Logo wenn offline
            return caches.match('/static/edge-logo.svg');
          });
        });
      })
    );
    return;
  }

  // Statische Assets (CSS, JS, SVG): Cache-First
  if (url.includes('/static/')) {
    e.respondWith(
      caches.match(e.request).then(function(cached) {
        return cached || fetch(e.request).then(function(resp) {
          if (resp.status === 200 && e.request.method === 'GET') {
            var clone = resp.clone();
            caches.open(CACHE_NAME).then(function(cache) {
              cache.put(e.request, clone);
            });
          }
          return resp;
        });
      })
    );
    return;
  }

  // API Endpoints: Network-First mit Cache Fallback (5min stale OK)
  if (url.includes('/api/')) {
    e.respondWith(
      fetch(e.request).then(function(resp) {
        if (resp.status === 200 && e.request.method === 'GET') {
          var clone = resp.clone();
          caches.open(CACHE_NAME).then(function(cache) {
            cache.put(e.request, clone);
          });
        }
        return resp;
      }).catch(function() {
        return caches.match(e.request);
      })
    );
    return;
  }

  // HTML Seiten: Stale-While-Revalidate (zeigt Cache SOFORT, updated im Hintergrund)
  if (e.request.mode === 'navigate' || e.request.headers.get('accept').includes('text/html')) {
    e.respondWith(
      caches.open(CACHE_NAME).then(function(cache) {
        return cache.match(e.request).then(function(cached) {
          var networkFetch = fetch(e.request).then(function(resp) {
            if (resp.status === 200) {
              cache.put(e.request, resp.clone());
            }
            return resp;
          });
          // Zeige gecachte Version SOFORT, update im Hintergrund
          return cached || networkFetch;
        });
      })
    );
    return;
  }

  // Default: Network-First
  e.respondWith(
    fetch(e.request).then(function(resp) {
      if (e.request.method === 'GET' && resp.status === 200) {
        var clone = resp.clone();
        caches.open(CACHE_NAME).then(function(cache) {
          cache.put(e.request, clone);
        });
      }
      return resp;
    }).catch(function() {
      return caches.match(e.request);
    })
  );
});
