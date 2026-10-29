/* ЕЩЁ НЕ КОНЕЦ — photo light.
   When the camera rests, the frame is re-lit by a path tracer: real bounce light off the walls, soft area shadows,
   contact darkening under the chair and the stone. The view through the window stays the live film from the room
   renderer (the path tracer leaves those pixels empty), and the result goes through the same depth of field, tone
   map and grain as the live picture, so the switch reads as the light settling, not as a different image. */
import * as THREE from 'three';
import { WebGLPathTracer } from 'three-gpu-pathtracer';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

const VS = `varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;

export function createPhoto({ renderer, scene, view, makeBokeh, grainShader, maxSamples = 1500, textureSize = 2048, maxPixels = 2.4e6, onProgress = () => {} }) {
  const pt = new WebGLPathTracer(renderer);
  Object.assign(pt, { renderToCanvas: false, rasterizeScene: false, renderDelay: 0, minSamples: 1, fadeDuration: 0, dynamicLowRes: false });
  pt.tiles.set(1, 1);
  pt.bounces = 7; pt.transmissiveBounces = 8; pt.filterGlossyFactor = .35; pt.multipleImportanceSampling = true;
  pt.textureSize.set(textureSize, textureSize);

  /* what the window shows: the room renderer's own sky, drawn alone */
  const skyRT = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType });
  const skyScene = new THREE.Scene();

  /* composite: path traced room (premultiplied by its own coverage) over the live sky, softened while samples are few */
  const mix = new ShaderPass({
    uniforms: { tPT: { value: null }, tSky: { value: null }, uK: { value: 1 }, uTex: { value: new THREE.Vector2(1, 1) }, uDn: { value: 1 }, uFF: { value: 1 } },
    vertexShader: VS,
    fragmentShader: `uniform sampler2D tPT,tSky;uniform float uK,uDn,uFF;uniform vec2 uTex;varying vec2 vUv;
      float lum(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
      void main(){vec4 p=texture2D(tPT,vUv);
        if(uFF>.5){ // lone bright specks (light caught at a grazing angle) are pulled down to their neighbourhood
          float mx=0.;for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++){if(x==0&&y==0)continue;mx=max(mx,lum(texture2D(tPT,vUv+vec2(float(x),float(y))*uTex).rgb));}
          float l=lum(p.rgb),cap=mx*1.6+.01;if(l>cap)p.rgb*=cap/l;}
        if(uDn>.001){ // edge-aware smoothing of the remaining noise; fades out as the picture converges
          vec3 acc=p.rgb;float ws=1.,l0=log(lum(p.rgb)+.002);
          for(int y=-2;y<=2;y++)for(int x=-2;x<=2;x++){if(x==0&&y==0)continue;vec2 o=vec2(float(x),float(y));
            vec4 q=texture2D(tPT,vUv+o*uTex*1.5);float d=log(lum(q.rgb)+.002)-l0;
            float w=exp(-dot(o,o)/(2.*2.2*2.2))*exp(-d*d/(2.*uDn*uDn))*step(.5,q.a)*step(.5,p.a);acc+=q.rgb*w;ws+=w;}
          p.rgb=acc/ws;}
        vec3 s=texture2D(tSky,vUv).rgb;
        gl_FragColor=vec4(p.rgb*uK+s*(1.-clamp(p.a,0.,1.)),1.);}`,
  });
  const comp = new EffectComposer(renderer);
  comp.addPass(mix);
  const bokeh = makeBokeh ? makeBokeh() : null;
  if (bokeh) {
    // the room is frozen while it is traced: its depth for the depth of field is drawn once, not every frame
    const full = bokeh.render.bind(bokeh), empty = new THREE.Scene();
    bokeh.render = (r, w, rd) => {
      if (!state.depthDone) { full(r, w, rd); state.depthDone = true; return; }
      const real = bokeh.scene, rt = bokeh.renderTargetDepth, prevC = r.autoClear;
      // reuse the stored depth: point the pass at an empty scene and keep its depth target from being cleared
      bokeh.scene = empty; bokeh.renderTargetDepth = state.dummyDepth || (state.dummyDepth = new THREE.WebGLRenderTarget(1, 1));
      bokeh.uniforms.tDepth.value = rt.texture;
      full(r, w, rd);
      bokeh.scene = real; bokeh.renderTargetDepth = rt; bokeh.uniforms.tDepth.value = rt.texture; r.autoClear = prevC;
    };
    comp.addPass(bokeh);
  }
  comp.addPass(new OutputPass());
  const grain = new ShaderPass(grainShader); comp.addPass(grain);

  /* exposure match: the path traced light is scaled so the room keeps the brightness of the live picture */
  const CW = 96;
  const smallR = new THREE.WebGLRenderTarget(CW, 54, { type: THREE.FloatType });
  const smallP = new THREE.WebGLRenderTarget(CW, 54, { type: THREE.FloatType });
  const down = new FullScreenQuad(new THREE.ShaderMaterial({
    uniforms: { map: { value: null }, uTex: { value: new THREE.Vector2() } }, vertexShader: VS,
    fragmentShader: `uniform sampler2D map;uniform vec2 uTex;varying vec2 vUv;void main(){vec4 a=vec4(0);for(int y=0;y<4;y++)for(int x=0;x<4;x++)a+=texture2D(map,vUv+(vec2(float(x),float(y))-1.5)*uTex*.25);gl_FragColor=a/16.;}`,
  }));
  function calibrate() {
    const W = CW, H = Math.max(8, Math.round(CW / view.aspect));
    smallR.setSize(W, H); smallP.setSize(W, H);
    const prevT = renderer.getRenderTarget();
    renderer.setRenderTarget(smallR); renderer.clear(); renderer.render(scene, view);
    down.material.uniforms.map.value = pt.target.texture; down.material.uniforms.uTex.value.set(1 / W, 1 / H);
    renderer.setRenderTarget(smallP); down.render(renderer);
    renderer.setRenderTarget(prevT);
    const a = new Float32Array(W * H * 4), b = new Float32Array(W * H * 4);
    renderer.readRenderTargetPixels(smallR, 0, 0, W, H, a); renderer.readRenderTargetPixels(smallP, 0, 0, W, H, b);
    const r = [];
    for (let i = 0; i < W * H; i++) {
      const o = i * 4; if (b[o + 3] < .99) continue;
      const lr = .2126 * a[o] + .7152 * a[o + 1] + .0722 * a[o + 2], lp = .2126 * b[o] + .7152 * b[o + 1] + .0722 * b[o + 2];
      if (lr > 1e-4 && lp > 1e-5 && isFinite(lr / lp)) r.push(Math.log(lr / lp));
    }
    if (r.length < 40) return null;
    r.sort((x, y) => x - y);
    // middle half of the ratios: robust to the window, to noise and to the places the two lights genuinely disagree
    let s = 0, n = 0; for (let i = Math.floor(r.length * .25); i < Math.floor(r.length * .75); i++) { s += r[i]; n++; }
    return Math.min(8, Math.max(.15, Math.exp(s / n)));
  }

  /* the scene as the path tracer should see it */
  const proxies = new THREE.Group(); proxies.name = 'pt-proxies'; proxies.visible = false; scene.add(proxies);
  const instPool = new Map(), matPool = new Map(); const splitPool = new Map(), flatPool = new Map(), alphaPool = new Map(); window.__ptAlphaPool = alphaPool;
  const _m = new THREE.Matrix4(), _c = new THREE.Color();
  const tintMat = (m, c) => {
    const k = m.uuid + ':' + c.getHexString(); let t = matPool.get(k);
    if (!t) { t = m.clone(); t.onBeforeCompile = () => {}; t.color.multiply(c); matPool.set(k, t); }
    return t;
  };
  let noteMats = null;
  // the tracer ignores draw ranges: a geometry drawn only in part gets an index that covers just that part
  const rangePool = new Map();
  const ranged = (geo, key) => {
    const dr = geo.drawRange, n = geo.index ? geo.index.count : geo.attributes.position.count;
    if (dr.start === 0 && dr.count >= n) return geo;
    const s0 = dr.start, cnt = Math.min(dr.count, n - s0);
    let r = rangePool.get(key);
    if (!r || r.src !== geo || r.cnt !== cnt) {
      const g = new THREE.BufferGeometry(); r = { src: geo, cnt, g, idx: new Uint32Array(cnt) };
      g.setIndex(new THREE.BufferAttribute(r.idx, 1)); rangePool.set(key, r);
    }
    for (const k in geo.attributes) if (r.g.attributes[k] !== geo.attributes[k]) r.g.setAttribute(k, geo.attributes[k]);
    for (let i = 0; i < cnt; i++) r.idx[i] = geo.index ? geo.index.getX(s0 + i) : s0 + i;
    r.g.index.needsUpdate = true;
    return r.g;
  };
  function prepare() {
    const hidden = [];
    const hide = o => { o.visible = false; hidden.push(o); };
    proxies.clear();
    const used = new Set();
    scene.traverseVisible(o => {
      if (o === proxies) return;
      if (o.userData.ptSky || o.isPoints || o.isLine || o.isSprite) { hide(o); return; }
      if (o.isLight) { if (!o.userData.ptKeep && !o.isAmbientLight) hide(o); return; }
      if (!o.isMesh) return;
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      if (mats.some(m => !m || m.colorWrite === false || m.visible === false || m.isShaderMaterial || m.isPointsMaterial || (m.isMeshBasicMaterial && (!m.depthWrite || m.transparent)))) { hide(o); return; }
      if (o.isInstancedMesh) {
        hide(o);
        let pool = instPool.get(o.uuid); if (!pool) { pool = []; instPool.set(o.uuid, pool); }
        for (let i = 0; i < o.count; i++) {
          let p = pool[i];
          if (!p) { p = new THREE.Mesh(o.geometry, o.material); p.matrixAutoUpdate = false; pool[i] = p; }
          o.getMatrixAt(i, _m); p.matrix.multiplyMatrices(o.matrixWorld, _m);
          if (o.instanceColor) { o.getColorAt(i, _c); _c.setRGB(Math.round(_c.r * 40) / 40, Math.round(_c.g * 40) / 40, Math.round(_c.b * 40) / 40); p.material = tintMat(o.material, _c); }
          proxies.add(p); used.add(p);
        }
        return;
      }
      if (!Array.isArray(o.material) && o.material.alphaMap && o.material.alphaMap.channel && o.geometry.attributes.uv && !o.isInstancedMesh) {
        // the torn holes and the hidden run of the roll live on a second uv set, which the tracer cannot read:
        // the cut-out is redrawn once into the first uv set and handed over as an ordinary alpha map
        hide(o);
        const am = o.material.alphaMap;
        let rec = alphaPool.get(o.uuid);
        if (!rec || rec.ver !== am.version || rec.geo !== o.geometry) {
          if (rec) rec.rt.dispose();
          const geo = o.geometry, UV = geo.attributes.uv, UV1 = geo.attributes[am.channel === 1 ? 'uv1' : 'uv' + am.channel] || geo.attributes.uv2;
          let u0 = 1e9, v0 = 1e9, u1 = -1e9, v1 = -1e9;
          for (let i = 0; i < UV.count; i++) { const u = UV.getX(i), v = UV.getY(i); u0 = Math.min(u0, u); v0 = Math.min(v0, v); u1 = Math.max(u1, u); v1 = Math.max(v1, v); }
          const su = Math.max(1e-4, u1 - u0), sv = Math.max(1e-4, v1 - v0);
          const W = 1024, H = Math.min(8192, Math.max(256, Math.round(1024 * sv / su / 64) * 64 || 1024));
          const rt = new THREE.WebGLRenderTarget(W, H, { generateMipmaps: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter });
          am.updateMatrix();
          const bake = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.ShaderMaterial({
            side: THREE.DoubleSide, depthTest: false, depthWrite: false,
            uniforms: { A: { value: am }, uM: { value: am.matrix }, uR: { value: new THREE.Vector4(u0, v0, su, sv) } },
            vertexShader: `attribute vec2 uvA;uniform mat3 uM;uniform vec4 uR;varying vec2 vA;void main(){vA=(uM*vec3(uvA,1.)).xy;gl_Position=vec4((uv-uR.xy)/uR.zw*2.-1.,0.,1.);}`,
            fragmentShader: `uniform sampler2D A;varying vec2 vA;void main(){float a=texture2D(A,clamp(vA,0.,1.)).g;gl_FragColor=vec4(a,a,a,1.);}`,
          }));
          bake.geometry.setAttribute('position', geo.attributes.position); bake.geometry.setAttribute('uv', UV); bake.geometry.setAttribute('uvA', UV1);
          if (geo.index) bake.geometry.setIndex(geo.index);
          bake.frustumCulled = false;
          const bs = new THREE.Scene(); bs.add(bake);
          const prevT = renderer.getRenderTarget(), prevC = renderer.getClearColor(new THREE.Color()), prevA = renderer.getClearAlpha();
          renderer.setRenderTarget(rt); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(bs, view);
          renderer.setRenderTarget(prevT); renderer.setClearColor(prevC, prevA);
          bake.geometry.dispose(); bake.material.dispose();
          const t = rt.texture; t.matrixAutoUpdate = true; t.repeat.set(1 / su, 1 / sv); t.offset.set(-u0 / su, -v0 / sv);
          const m = o.material.clone(); m.onBeforeCompile = () => {}; m.alphaMap = t;
          const p = new THREE.Mesh(o.geometry, m); p.matrixAutoUpdate = false;
          rec = { rt, ver: am.version, geo: o.geometry, p }; alphaPool.set(o.uuid, rec);
        }
        // keep the copy's look in step with the live material
        rec.p.geometry = ranged(o.geometry, o.uuid + ':a');
        rec.p.material.color.copy(o.material.color); rec.p.matrix.copy(o.matrixWorld); proxies.add(rec.p);
        return;
      }
      if (!Array.isArray(o.material) && o.material.displacementMap && !o.isInstancedMesh) {
        // the floorboards are raised by a height map in the live renderer; the tracer gets that relief baked into a copy
        hide(o);
        let p = flatPool.get(o.uuid);
        if (!p) {
          const m = o.material, dmap = m.displacementMap, img = dmap.image;
          const cv = document.createElement('canvas'); cv.width = img.width; cv.height = img.height;
          const cx = cv.getContext('2d'); cx.drawImage(img, 0, 0); const px = cx.getImageData(0, 0, cv.width, cv.height).data, W = cv.width, H = cv.height;
          const hAt = (u, v) => { // bilinear, repeating, three's uv convention (v up, image rows down unless flipY is off)
            u = u - Math.floor(u); v = v - Math.floor(v); if (dmap.flipY) v = 1 - v;
            const x = u * W - .5, y = v * H - .5, x0 = Math.floor(x), y0 = Math.floor(y), tx = x - x0, ty = y - y0;
            const g = (i, j) => px[((((j % H) + H) % H) * W + (((i % W) + W) % W)) * 4] / 255;
            return (g(x0, y0) * (1 - tx) + g(x0 + 1, y0) * tx) * (1 - ty) + (g(x0, y0 + 1) * (1 - tx) + g(x0 + 1, y0 + 1) * tx) * ty;
          };
          dmap.updateMatrix(); const tm = dmap.matrix.elements;
          const src = o.geometry, P = src.attributes.position, N = src.attributes.normal, UV = src.attributes.uv;
          const g = new THREE.BufferGeometry(); for (const k in src.attributes) if (k !== 'position') g.setAttribute(k, src.attributes[k]);
          if (src.index) g.setIndex(src.index);
          const out = new Float32Array(P.count * 3);
          for (let i = 0; i < P.count; i++) {
            const u0 = UV.getX(i), v0 = UV.getY(i), u = tm[0] * u0 + tm[3] * v0 + tm[6], v = tm[1] * u0 + tm[4] * v0 + tm[7];
            const d = hAt(u, v) * m.displacementScale + m.displacementBias;
            out[i * 3] = P.getX(i) + N.getX(i) * d; out[i * 3 + 1] = P.getY(i) + N.getY(i) * d; out[i * 3 + 2] = P.getZ(i) + N.getZ(i) * d;
          }
          g.setAttribute('position', new THREE.BufferAttribute(out, 3)); g.computeVertexNormals();
          p = new THREE.Mesh(g, o.material); p.matrixAutoUpdate = false; flatPool.set(o.uuid, p);
        }
        p.matrix.copy(o.matrixWorld); proxies.add(p);
        return;
      }
      if (!Array.isArray(o.material) && o.material.polygonOffset && !o.isInstancedMesh) {
        // pieces the live renderer keeps on top of a surface with a depth bias (ties on the paper, paper on the floor):
        // the tracer has no bias, so the piece is pushed out along its own normals by a millimetre and a half
        hide(o);
        const src = o.geometry, P = src.attributes.position;
        if (!src.attributes.normal) src.computeVertexNormals();
        const N = src.attributes.normal;
        let p = flatPool.get(o.uuid);
        if (!p || p.userData.src !== src || p.geometry.attributes.position.count !== P.count) {
          const g = new THREE.BufferGeometry();
          for (const k in src.attributes) if (k !== 'position') g.setAttribute(k, src.attributes[k]);
          g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(P.count * 3), 3));
          if (src.index) g.setIndex(src.index);
          p = new THREE.Mesh(g, o.material); p.matrixAutoUpdate = false; p.userData.src = src; flatPool.set(o.uuid, p);
        }
        const out = p.geometry.attributes.position, d = .0015 * Math.max(1, Math.abs(o.material.polygonOffsetUnits || 0) / 1.5) / Math.max(1e-6, o.matrixWorld.getMaxScaleOnAxis());
        for (let i = 0; i < P.count; i++) out.setXYZ(i, P.getX(i) + N.getX(i) * d, P.getY(i) + N.getY(i) * d, P.getZ(i) + N.getZ(i) * d);
        out.needsUpdate = true; p.geometry.setDrawRange(src.drawRange.start, src.drawRange.count);
        p.matrix.copy(o.matrixWorld); proxies.add(p);
        return;
      }
      if (!Array.isArray(o.material) && (o.material.isMeshBasicMaterial || (!o.material.depthWrite && o.material.transparent && !o.material.isMeshPhysicalMaterial))) {
        // unlit pieces (the black ties) become plain matte surfaces; decals drawn flat on the floor are lifted a hair
        // off it, since the tracer has no draw order to settle two surfaces in one plane
        hide(o);
        let p = flatPool.get(o.uuid);
        if (!p) {
          let m = o.material;
          if (m.isMeshBasicMaterial) m = new THREE.MeshStandardMaterial({ color: m.color, map: m.map, alphaMap: m.alphaMap, alphaTest: m.alphaTest, transparent: m.transparent, opacity: m.opacity, side: m.side, roughness: .7 });
          p = new THREE.Mesh(o.geometry, m); p.matrixAutoUpdate = false; flatPool.set(o.uuid, p);
        }
        p.geometry = o.geometry; p.matrix.copy(o.matrixWorld);
        if (!o.material.isMeshBasicMaterial) p.matrix.elements[13] += .0018;
        proxies.add(p);
        return;
      }
      if (!Array.isArray(o.material) && ranged(o.geometry, o.uuid) !== o.geometry) {
        hide(o);
        let p = flatPool.get(o.uuid + ':r'); if (!p) { p = new THREE.Mesh(undefined, o.material); p.matrixAutoUpdate = false; flatPool.set(o.uuid + ':r', p); }
        p.geometry = ranged(o.geometry, o.uuid); p.material = o.material; p.matrix.copy(o.matrixWorld); proxies.add(p);
        return;
      }
      if (Array.isArray(o.material)) {
        // the tracer gives each mesh one material: split a multi-material mesh into one piece per group
        hide(o);
        const geo = o.geometry, n = geo.index ? geo.index.count : geo.attributes.position.count;
        let parts = splitPool.get(o.uuid);
        if (!parts || parts.geo !== geo) {
          parts = { geo, list: [] };
          for (const gr of (geo.groups.length ? geo.groups : [{ start: 0, count: n, materialIndex: 0 }])) {
            const g = new THREE.BufferGeometry();
            for (const k in geo.attributes) g.setAttribute(k, geo.attributes[k]);
            const cnt = Math.min(gr.count, n - gr.start), idx = new Uint32Array(cnt);
            for (let k = 0; k < cnt; k++) idx[k] = geo.index ? geo.index.getX(gr.start + k) : gr.start + k;
            g.setIndex(new THREE.BufferAttribute(idx, 1));
            const m = new THREE.Mesh(g, o.material[gr.materialIndex] || o.material[0]); m.matrixAutoUpdate = false;
            parts.list.push(m);
          }
          splitPool.set(o.uuid, parts);
        }
        parts.list.forEach(m => { m.matrix.copy(o.matrixWorld); proxies.add(m); });
        return;
      }
      if (o.userData.ptFace) {
        // the note: text only on the face that carries it, plain paper behind
        hide(o);
        const m0 = o.material;
        if (!noteMats) {
          const plain = new THREE.MeshStandardMaterial({ color: new THREE.Color(.93, .925, .9), roughness: m0.roughness });
          noteMats = { text: m0.clone(), plain };
          noteMats.text.onBeforeCompile = () => {};
        }
        const f = o.userData.ptFace() > 0;
        noteMats.text.side = f ? THREE.FrontSide : THREE.BackSide; noteMats.plain.side = f ? THREE.BackSide : THREE.FrontSide;
        for (const mm of [noteMats.text, noteMats.plain]) {
          const p = new THREE.Mesh(o.geometry, mm); p.matrixAutoUpdate = false; p.matrix.copy(o.matrixWorld); proxies.add(p);
        }
      }
    });
    proxies.visible = true;
    return hidden;
  }

  const state = { active: false, shown: false, k: 1, kT: 1, cal: [], building: false };
  let bg0 = null, env0 = null;

  const TINT = new THREE.Color(.86, .93, 1.0), SKYC = new THREE.Color('#dfe7ee');
  function start() {
    state.building = true;
    const hidden = prepare();
    const saved = new Map(), savedL = [];
    const dbg = window.__ptDbg, savedE = [], savedT = [];
    const touch = m => { if (m && m.color && !saved.has(m)) { saved.set(m, m.color.clone()); m.color.multiply(TINT).multiplyScalar(.9);
      // the tracer reads every map through the first uv set; a map laid on another set (the torn holes) is left out
      for (const k of ['alphaMap', 'map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap']) if (m[k] && m[k].channel) { savedT.push([m, k, m[k]]); m[k] = null; }
      if (dbg && m.emissive) { savedE.push([m, m.emissive.clone(), m.emissiveMap, m.emissiveIntensity]); m.emissive.setRGB(1, 1, 1); m.emissiveMap = m.map; m.emissiveIntensity = 1; m.color.setRGB(0, 0, 0); } } };
    scene.traverseVisible(o => { if (o.isMesh) (Array.isArray(o.material) ? o.material : [o.material]).forEach(touch);
      if (o.isRectAreaLight) { savedL.push([o, o.color.clone()]); o.color.copy(SKYC); } });
    bg0 = scene.background; env0 = scene.environment; scene.background = null; scene.environment = null;
    view.updateMatrixWorld();
    try { pt.setScene(scene, view); }
    finally {
      scene.background = bg0; scene.environment = env0;
      hidden.forEach(o => o.visible = true); proxies.visible = false;
      savedT.forEach(([m, k, t]) => { m[k] = t; }); savedE.forEach(([m, e, em, ei]) => { m.emissive.copy(e); m.emissiveMap = em; m.emissiveIntensity = ei; }); saved.forEach((c, m) => m.color.copy(c)); savedL.forEach(([l, c]) => l.color.copy(c));
      state.building = false;
    }
    pt.pausePathTracing = false;
    // the room is sampled at no more than ~2.4 million pixels and scaled up
    const sz = renderer.getDrawingBufferSize(new THREE.Vector2());
    pt.renderScale = Math.min(1, Math.sqrt(maxPixels / (sz.x * sz.y)));
    state.active = true; state.shown = false; state.depthDone = false; state.cal = [6, 24, 96]; state.k = state.kT = 1;
    onProgress(0);
  }
  function prewarm() {
    if (state.active || state.warm) return; state.warm = true;
    start(); state.active = false; pt.pausePathTracing = true; onProgress(-1);
    pt.renderSample(); // kicks off the asynchronous compile
  }
  function stop() {
    if (!state.active) return;
    state.active = false; state.shown = false; pt.pausePathTracing = true; pt.reset();
    onProgress(-1);
  }
  let skyMesh = null;
  scene.traverse(o => { if (o.userData.ptSky) skyMesh = o; });
  function renderSky(w, h) {
    if (skyRT.width !== w || skyRT.height !== h) skyRT.setSize(w, h);
    const parent = skyMesh.parent; skyScene.add(skyMesh);
    const prevT = renderer.getRenderTarget();
    renderer.setRenderTarget(skyRT); renderer.setClearColor(0x000000, 1); renderer.clear(); renderer.render(skyScene, view);
    renderer.setRenderTarget(prevT);
    parent.add(skyMesh);
  }

  /* one step: more samples; returns true once the traced picture is the one on screen */
  function step({ focus, time }) {
    if (!state.active) return false;
    if (pt.samples < maxSamples) pt.renderSample();
    else onProgress(1);
    const n = pt.samples;
    if (state.cal.length && n >= state.cal[0]) {
      state.cal.shift(); const k = calibrate(); if (k) { state.kT = k; if (!state.shown) state.k = k; }
    }
    state.k += (state.kT - state.k) * .08;
    if (n < 6) { onProgress(n / maxSamples); return false; }
    const sz = renderer.getDrawingBufferSize(new THREE.Vector2());
    renderSky(sz.x, sz.y);
    mix.uniforms.tPT.value = pt.target.texture; mix.uniforms.tSky.value = skyRT.texture; mix.uniforms.uK.value = state.k;
    mix.uniforms.uTex.value.set(1 / pt.target.width, 1 / pt.target.height);
    mix.uniforms.uFF.value = n < 800 ? 1 : 0;
    mix.uniforms.uDn.value = n >= 400 ? 0 : .55 * Math.sqrt(8 / Math.max(8, n)) * (1 - n / 400);
    if (bokeh && focus !== undefined) bokeh.uniforms.focus.value = focus;
    grain.uniforms.uTime.value = time;
    comp.render();
    state.shown = true;
    onProgress(Math.min(1, n / maxSamples));
    return true;
  }
  function setSize(w, h) { comp.setSize(w, h); state.depthDone = false; }
  function setBokeh(on) { if (bokeh) bokeh.enabled = on; }

  return {
    pt, start, stop, step, setSize, setBokeh,
    prewarm, get ready() { return state.active && !state.shown && pt.samples >= 5; },
    get active() { return state.active; }, get shown() { return state.shown; }, get samples() { return pt.samples; },
    get compiling() { return pt.isCompiling; },
  };
}
