# Route Replay Feature — Implementation Context

> Saved 2026-02-02. Resume implementation from this file.

## What We're Building

Add a "Walking Network" layer + interactive route replay to `docs/viewer.html`:
1. Display route graph edges (5622 edges, 3856 nodes) colored by PCI/shade/LST
2. User clicks start/end points on map → Dijkstra shortest path computed client-side
3. "Replay" = animate the Deck.GL camera along the route at street-level view (zoom 18, pitch 70°, bearing facing travel direction)

## Architecture Understanding

### Viewer Stack
- **NOT Three.js** — uses **Deck.GL 8.9.0** + **MapLibre-GL 2.4.0**
- Single-file: `docs/viewer.html` (1726 lines, 84KB)
- CSS in separate `docs/viewer.css` (760 lines)
- No external JS files — all inline
- Layer system: accordion sidebar with toggles, `buildLayers()` returns array of Deck.GL layers, `rebuildDeck()` calls `deckgl.setProps()`

### Existing View Presets (line ~1264)
- Overview: zoom 15, pitch 55°, bearing -20°
- Street: zoom 18, pitch 70°, bearing 30° ← this is the "street view" angle for replay

### Data Loading Pattern
- `area.json` manifest maps layer keys → file names
- `dataKeyMap` in JS maps toggle IDs → area.json keys
- `ensureDataLoaded(key)` loads lazily into `CACHE[key]`
- `route_graph` is ALREADY registered in area.json (line 45) but NOT yet loaded by viewer.html

### Route Graph Data (`docs/data/al_karama/route_graph.json`, 298KB)
```
{
  "nodes": [[lat, lon], ...],                    // 3856 nodes
  "edges": [[u, v, len, pci, lst, shade, fi], ...], // 5622 edges
  "edge_keys": ["u", "v", "len", "pci", "lst", "shade", "fi"],
  "default_pci": 0.4269,
  "stats": { "node_count": 3856, "edge_count": 5622, "pci_coverage": 0.268 }
}
```
- `fi` = index into `streets.geojson` features for real polyline geometry
- `pci` = Pedestrian Comfort Index (can be null for unmeasured edges)
- `lst` = Land Surface Temperature in °C
- `shade` = shade fraction (0-1)

### Code to Port from viewer_2d.html
Already working in `docs/viewer_2d.html`:
- `buildRouteAdjacency(graph)` — lines 1470-1516: builds adjacency list with `distW` and `comfW` per edge
- `dijkstra(adj, start, end, weightKey)` — lines 1518-1568: typed-array Dijkstra with binary min-heap
- `heapPush(h, item)` — lines 1571-1580
- `heapPop(h)` — lines 1582-1598
- `snapToNearestNode(latlng)` — lines 1600-1612: linear scan, sub-ms on 3856 nodes

**Key adaptation**: viewer_2d uses Leaflet's `{lat, lng}` format. Deck.GL `onClick` provides `info.coordinate` as `[lng, lat]`. Adjust `snapToNearestNode()` accordingly.

### Comfort Weight Formula (from buildRouteAdjacency)
```
comfort = 0.40 * pci + 0.35 * shade + 0.25 * lstComfort
comfW = len * (3 - 2 * comfort)  // multiplier range 1.0 (comfort=1) to 3.0 (comfort=0)
```
Unmeasured edges get comfort=0.25 (pessimistic).

## Full Implementation Plan

### State Variables (add ~line 309)
```javascript
var routeGraph = null;         // raw route_graph.json
var routeAdj = null;           // adjacency list
var routeMode = false;         // click-to-route active
var routeOriginNode = null;    // node index
var routeDestNode = null;      // node index
var routeResult = null;        // {path, edges, cost}
var routeEdgePaths = null;     // array of {path, pci, len} for highlight
var routeCoords = null;        // continuous [lng, lat] polyline
var routeAnimating = false;
var routeAnimPaused = false;
var routeAnimProgress = 0;     // 0..1
var routeAnimSpeed = 1;        // 0.5x/1x/2x/4x
var routeAnimRafId = null;
var routeAnimTotalDist = 0;
var routeAnimSamples = null;   // [{lat, lng, bearing, dist}, ...]
var routeAnimLastTime = 0;
var routeGraphColorMode = 'pci'; // 'pci' | 'shade' | 'lst'
var routeWeightMode = 'comfort'; // 'comfort' | 'distance'
var WALK_SPEED_MPS = 1.4;
```
Add `'route-graph': false` to `layerState` object.

### HTML Changes — Sidebar (inside "Mobility & Network" accordion, after comfort layer ~line 172)
- Layer toggle for "Walking Network" + info button
- Layer options: PCI/shade/LST radio buttons
- Route controls div: "Set Route" button, "Clear" button, weight mode radio (Comfort/Shortest), status text

### HTML Changes — Replay Bar (floating above time slider, ~line 284)
- Play/pause button (circle with ▶/⏸)
- Progress slider with filled track
- Speed dropdown (0.5x/1x/2x/4x)
- Distance readout (e.g. "320m / 850m")
- Exit button (✕)

### CSS Changes (viewer.css)
- `.route-action-btn` — cyan accent buttons
- `.route-replay-bar` — fixed bottom bar, dark glass background, above time slider
- `.replay-btn`, `.replay-progress-*` — playback controls
- `#map-canvas.route-picking { cursor: crosshair; }`

### JS — Data Loading
- Add to `dataKeyMap`: `'route-graph': 'route_graph'`
- In `onLayerToggle('route-graph')`: show/hide options, load data, build adjacency list
- Also ensure `streets` data is loaded (for edge geometry via `fi`)

### JS — Deck.GL Layers (in buildLayers())
1. **Route graph edges** (PathLayer): colored by routeGraphColorMode, 2px, opacity 0.7
   - Use `CACHE.streets.features[fi].geometry.coordinates` for real curves
   - Fallback to straight line between nodes if fi unavailable
2. **Route graph nodes** (ScatterplotLayer): gray dots at zoom >= 17
3. **Origin marker** (ScatterplotLayer): green, 8px, white stroke
4. **Destination marker** (ScatterplotLayer): red, 8px, white stroke
5. **Route highlight** (PathLayer): cyan, 6px, depthTest false
6. **Animated marker** (ScatterplotLayer): cyan+white, 10px, during replay only

### JS — Click Handling
- Add `onClick` to DeckGL constructor
- When routeMode: first click sets origin, second sets destination → `computeRoute()`
- `computeRoute()`: run Dijkstra, build edge paths (with direction correction), pre-compute animation samples, show replay bar

### JS — Camera Animation
- `buildAnimationSamples(coords, totalDist)`: sample every ~2m, compute bearing at each sample
- `routeAnimFrame(timestamp)`: requestAnimationFrame loop, advance by walk speed * dt
- Camera update via `deckgl.setProps({ initialViewState: { lng, lat, zoom:18, pitch:70, bearing, transitionDuration:80, transitionInterpolator: new deck.LinearInterpolator(['longitude','latitude','bearing']) } })`
- Controls: toggleRouteReplay(), onReplaySeek(), setReplaySpeed(), exitRouteReplay()

### JS — Helpers
- `haversineMeters(lat1, lon1, lat2, lon2)` — distance in meters
- `computeBearing(lat1, lon1, lat2, lon2)` — bearing in degrees
- `comfortColor(pci)` — green→yellow→red color ramp for PCI values
- Tooltip for route-graph-edges, legend entries, info popover

### Key Design Decisions
- **LinearInterpolator** (not FlyToInterpolator): avoids zoom bouncing between frames
- **80ms transitionDuration**: one-frame smoothing without lag
- **Edge geometry via `fi` index**: real road curves from streets.geojson
- **Bearing per-sample**: natural camera turns, smoothed by transition

## Verification Steps
1. `cd docs && python -m http.server 8000` → open viewer.html
2. Toggle "Walking Network" → edges appear colored by PCI
3. "Set Route" → click two points → green/red markers, cyan route
4. Toggle Comfort/Shortest → route recomputes
5. Play replay → camera animates at street level
6. Test pause, seek, speed, exit
7. Verify existing layers still work

## Files Summary
| File | Action |
|------|--------|
| `docs/viewer.html` | Add ~400 lines of JS + ~40 lines of HTML |
| `docs/viewer.css` | Add ~80 lines of CSS |
| `docs/viewer_2d.html` | READ ONLY — port Dijkstra code from lines 1470-1612 |
| `docs/data/al_karama/route_graph.json` | Already exists, no changes |
| `docs/data/al_karama/area.json` | Already has route_graph registered, no changes |
| `docs/data/al_karama/streets.geojson` | Already exists, used for edge geometry via fi |
