import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { mergeVertices, mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* ---------- contact-hardening soft shadows ---------- */
{ let ch=THREE.ShaderChunk.shadowmap_pars_fragment;
  const pcss=`
  float pcssShadow(sampler2D sm,vec2 uv,float zR,float K,vec2 size){
    float ang=rand(uv*size+fract(zR*97.0))*6.2831853,sn=sin(ang),cs=cos(ang);mat2 R=mat2(cs,-sn,sn,cs);
    float searchR=K*0.12+2.0/size.x,sumB=0.0,nB=0.0;
    for(int i=0;i<16;i++){float rr=sqrt((float(i)+.5)/16.0),a=float(i)*2.39996;vec2 o=R*vec2(cos(a),sin(a))*rr*searchR;
      float d=unpackRGBAToDepth(texture2D(sm,uv+o));if(d<zR){sumB+=d;nB+=1.0;}}
    if(nB<0.5)return 1.0;
    float zB=sumB/nB,r=clamp((zR-zB)*K,1.2/size.x,searchR),sum=0.0;
    for(int i=0;i<28;i++){float rr=sqrt((float(i)+.5)/28.0),a=float(i)*2.39996+1.1;vec2 o=R*vec2(cos(a),sin(a))*rr*r;
      sum+=step(zR,unpackRGBAToDepth(texture2D(sm,uv+o)));}
    return sum/28.0;}
  // the window as an area light: contact-hardening shadows for a perspective (spot) shadow map
  const float SPN=0.3,SPF=10.0,SPT=1.743,SPL=1.1;
  float pcssSpot(sampler2D sm,vec2 uv,float zR,vec2 size){
    float dR=-perspectiveDepthToViewZ(zR,SPN,SPF);
    float ang=rand(uv*size+fract(zR*97.0))*6.2831853,sn=sin(ang),cs=cos(ang);mat2 R=mat2(cs,-sn,sn,cs);
    float searchR=0.026,sumB=0.0,nB=0.0;
    for(int i=0;i<16;i++){float rr=sqrt((float(i)+.5)/16.0),a=float(i)*2.39996;vec2 o=R*vec2(cos(a),sin(a))*rr*searchR;
      float d=unpackRGBAToDepth(texture2D(sm,uv+o));if(d<zR){sumB+=-perspectiveDepthToViewZ(d,SPN,SPF);nB+=1.0;}}
    if(nB<0.5)return 1.0;
    float dB=sumB/nB,r=clamp(SPL*(dR-dB)/(dB*2.0*SPT*dR),1.5/size.x,searchR),sum=0.0;
    for(int i=0;i<28;i++){float rr=sqrt((float(i)+.5)/28.0),a=float(i)*2.39996+1.1;vec2 o=R*vec2(cos(a),sin(a))*rr*r;
      sum+=step(zR,unpackRGBAToDepth(texture2D(sm,uv+o)));}
    return sum/28.0;}
  float getShadow(`;
  ch=ch.replace('float getShadow(',pcss);
  ch=ch.replace('if ( frustumTest ) {','if ( frustumTest ) {\n if(shadowRadius>199.0){shadow=pcssSpot(shadowMap,shadowCoord.xy,shadowCoord.z,shadowMapSize);}else if(shadowRadius>99.0){shadow=pcssShadow(shadowMap,shadowCoord.xy,shadowCoord.z,shadowRadius-100.0,shadowMapSize);}else{');
  ch=ch.replace('\t\t#endif\n\t\t}\n\t\treturn shadow;','\t\t#endif\n\t\t}}\n\t\treturn shadow;');
  THREE.ShaderChunk.shadowmap_pars_fragment=ch; }

/* ---------- helpers ---------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function makeNoise(seed){
  const r=mulberry32(seed),p=new Uint16Array(512),v=new Float32Array(256);
  for(let i=0;i<256;i++){p[i]=i;v[i]=r()}
  for(let i=255;i>0;i--){const j=Math.floor(r()*(i+1));[p[i],p[j]]=[p[j],p[i]]}
  for(let i=0;i<256;i++)p[i+256]=p[i];
  function n(x,y){const xi=Math.floor(x),yi=Math.floor(y),xf=x-xi,yf=y-yi,X=xi&255,Y=yi&255;
    const a=v[p[p[X]+Y]],b=v[p[p[X+1]+Y]],c=v[p[p[X]+Y+1]],d=v[p[p[X+1]+Y+1]];
    const u=xf*xf*(3-2*xf),w=yf*yf*(3-2*yf);return a+(b-a)*u+(c-a)*w+(a-b-c+d)*u*w}
  function fbm(x,y,o){let s=0,amp=.5,f=1,nm=0;for(let i=0;i<o;i++){s+=amp*n(x*f+i*17.3,y*f+i*9.1);nm+=amp;amp*=.5;f*=2.03}return s/nm}
  return {n,fbm};
}
const sstep=(a,b,x)=>{const t=Math.min(1,Math.max(0,(x-a)/(b-a)));return t*t*(3-2*t)};
const mix=(a,b,t)=>[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];
const clamp255=v=>v<0?0:v>255?255:v;
function canvas(w,h){const c=document.createElement('canvas');c.width=w;c.height=h;return c}
const tick=()=>new Promise(r=>setTimeout(r,0));

/* ---------- procedural textures ---------- */
function plasterTex(wM,hM,seed,pal,ppm=200,maxW=1024){
  const W=Math.min(maxW,Math.round(wM*ppm)),H=Math.max(8,Math.round(W*hM/wM));
  const cm=canvas(W,H),cb=canvas(W,H),im=cm.getContext('2d').createImageData(W,H),ib=cb.getContext('2d').createImageData(W,H);
  const nz=makeNoise(seed);
  for(let y=0;y<H;y++){const Y=(1-y/H)*hM;
    for(let x=0;x<W;x++){const X=x/W*wM;
      const big=nz.fbm(X*.9,Y*.9,5),mid=nz.fbm(X*3+7,Y*3,4),hue=nz.fbm(X*.45+30,Y*.45+11,3);
      const fine=nz.n(X*45,Y*45)*.5+nz.n(X*110+3,Y*110)*.5;
      const streak=nz.fbm(X*22+5,Y*.7,3);
      let c=mix(pal.a,pal.b,sstep(.3,.7,hue));
      c=mix(c,pal.c,sstep(.5,.72,mid)*.5);
      const cloud=.84+.32*big;
      const m2=nz.fbm(X*2.6+40,Y*2.6,4)+(fine-.5)*.08;
      const light=sstep(.58,.68,m2);
      c=mix(c,pal.peel,light*.7);
      const e=nz.fbm(X*5+70,Y*5,4)+(fine-.5)*.1;
      const peel=sstep(pal.peelAt,pal.peelAt+.012,e);
      const rim=Math.max(0,sstep(pal.peelAt-.02,pal.peelAt,e)-peel);
      c=mix(c,pal.peel,peel*.85);
      let k=cloud*(1-rim*.1)*(.88+.24*fine)*(1-.08*sstep(.55,.75,streak));
      if(pal.dirt){k*=1-.2*(1-sstep(0,.6,Y))*(.5+mid)}
      const i=(y*W+x)*4;
      im.data[i]=clamp255(c[0]*k);im.data[i+1]=clamp255(c[1]*k);im.data[i+2]=clamp255(c[2]*k);im.data[i+3]=255;
      const b=255*((1-peel)*.55+fine*.3+(1-rim)*.15);
      ib.data[i]=ib.data[i+1]=ib.data[i+2]=b;ib.data[i+3]=255;
    }}
  cm.getContext('2d').putImageData(im,0,0);cb.getContext('2d').putImageData(ib,0,0);
  return {map:cm,bump:cb};
}

function floorTex(){
  const wM=4,dM=5.2,ppm=300,W=wM*ppm,H=Math.round(dM*ppm);
  const cm=canvas(W,H),cr=canvas(W,H),cb=canvas(W,H);
  const im=cm.getContext('2d').createImageData(W,H),ir=cr.getContext('2d').createImageData(W,H),ib=cb.getContext('2d').createImageData(W,H);
  const nz=makeNoise(77),r=mulberry32(5);
  const pw=.125,np=Math.ceil(wM/pw),planks=[];
  for(let i=0;i<np;i++){const segs=[];let z=-r()*2.5;
    while(z<dM){const L=1.3+r()*2.2;
      segs.push({z0:z,z1:z+L,tone:[96+r()*14,75+r()*10,56+r()*8],seed:r()*500,shift:(r()-.5)*.14});z+=L}
    planks.push(segs)}
  for(let y=0;y<H;y++){const Z=y/ppm;
    for(let x=0;x<W;x++){const X=x/ppm;
      const pi=Math.min(np-1,Math.floor(X/pw)),u=(X-pi*pw)/pw,segs=planks[pi];
      let s=segs[0];for(let j=0;j<segs.length;j++){if(Z>=segs[j].z0&&Z<segs[j].z1){s=segs[j];break}}
      const grain=nz.fbm(X*38+s.seed,Z*1.8,4),str=nz.n(X*170+s.seed,Z*5);
      let c=[s.tone[0],s.tone[1],s.tone[2]];
      let k=(.72+.55*grain)*(.9+.2*str)*(1+s.shift);
      const wornN=nz.fbm(X*1.5+100,Z*1.5,4)+(nz.n(X*14,Z*14)-.5)*.3;
      const worn=sstep(.54,.7,wornN);
      const spots=sstep(.7,.8,nz.fbm(X*5+3,Z*5,3)+(str-.5)*.15);
      c=[c[0]*k,c[1]*k,c[2]*k];
      c=mix(c,[150,128,112],worn*.45);
      c=mix(c,[196,184,170],spots*.55);
      const edge=Math.min(u,1-u)*pw,endD=Math.min(Z-s.z0,s.z1-Z);
      let g=1;if(edge<.003)g=.35;else if(edge<.009)g=.8;
      if(endD<.003)g=Math.min(g,.4);
      c=[c[0]*g,c[1]*g,c[2]*g];
      const i=(y*W+x)*4;
      im.data[i]=clamp255(c[0]);im.data[i+1]=clamp255(c[1]);im.data[i+2]=clamp255(c[2]);im.data[i+3]=255;
      const rough=255*Math.min(1,.62+worn*.25+spots*.3+(1-g)*.3);
      ir.data[i]=ir.data[i+1]=ir.data[i+2]=rough;ir.data[i+3]=255;
      const b=255*(g*.75+grain*.25);ib.data[i]=ib.data[i+1]=ib.data[i+2]=b;ib.data[i+3]=255;
    }}
  const ctx=cm.getContext('2d');ctx.putImageData(im,0,0);
  cr.getContext('2d').putImageData(ir,0,0);cb.getContext('2d').putImageData(ib,0,0);
  // scratches and scuffs
  ctx.lineCap='round';
  for(let i=0;i<900;i++){const x=r()*W,y=r()*H,len=10+r()*90,a=(r()<.6?Math.PI/2:r()*Math.PI)+(r()-.5)*.5;
    ctx.strokeStyle=`rgba(${205+r()*30},${195+r()*25},${180+r()*25},${.06+r()*.16})`;ctx.lineWidth=.6+r()*1.4;
    ctx.beginPath();ctx.moveTo(x,y);ctx.quadraticCurveTo(x+Math.cos(a)*len*.5+(r()-.5)*8,y+Math.sin(a)*len*.5,x+Math.cos(a)*len,y+Math.sin(a)*len);ctx.stroke()}
  for(let i=0;i<120;i++){const x=r()*W,y=r()*H,rad=6+r()*28;
    const gr=ctx.createRadialGradient(x,y,0,x,y,rad);gr.addColorStop(0,`rgba(210,200,188,${.05+r()*.12})`);gr.addColorStop(1,'rgba(210,200,188,0)');
    ctx.fillStyle=gr;ctx.beginPath();ctx.ellipse(x,y,rad,rad*(.4+r()),r()*3,0,7);ctx.fill()}
  return {map:cm,rough:cr,bump:cb};
}

function doorTex(img){
  const ppm=400,wM=.8,hM=2.15,W=Math.round(wM*ppm),H=Math.round(hM*ppm);
  const cm=canvas(W,H),cb=canvas(W,H),im=cm.getContext('2d').createImageData(W,H);
  const nz=makeNoise(31);
  for(let y=0;y<H;y++){const Y=y/ppm;for(let x=0;x<W;x++){const X=x/ppm;
    const grain=nz.fbm(X*28,Y*1.4,5),fine=nz.n(X*240,Y*4);
    let c=[122,103,84];let k=(.68+.6*grain)*(.88+.24*fine);c=[c[0]*k,c[1]*k,c[2]*k];
    const w=sstep(.48,.66,nz.fbm(X*3+9,Y*2.4,4)+(fine-.5)*.2);c=mix(c,[182,174,160],w*.75);
    if(nz.n(X*220+40,Y*2.5)>.9)c=[c[0]*.6,c[1]*.6,c[2]*.6];
    const i=(y*W+x)*4;im.data[i]=clamp255(c[0]);im.data[i+1]=clamp255(c[1]);im.data[i+2]=clamp255(c[2]);im.data[i+3]=255}}
  const ctx=cm.getContext('2d');ctx.putImageData(im,0,0);if(img){for(let y=0;y<H;y+=W)ctx.drawImage(img,0,y,W,W)}
  const b=cb.getContext('2d');b.fillStyle='#909090';b.fillRect(0,0,W,H);
  const P=(x,y,w,h)=>[x*ppm,y*ppm,w*ppm,h*ppm];
  const panels=[P(.11,.12,.26,.46),P(.43,.12,.26,.46),P(.11,.72,.58,.16),P(.11,.98,.26,.62),P(.43,.98,.26,.62),P(.11,1.72,.58,.3)];
  for(const [x,y,w,h] of panels){
    ctx.fillStyle='rgba(40,28,18,.16)';ctx.fillRect(x,y,w,h);
    const bv=11;
    ctx.fillStyle='rgba(25,16,8,.42)';ctx.fillRect(x,y,w,bv*.5);ctx.fillRect(x,y,bv*.5,h);
    ctx.fillStyle='rgba(230,220,200,.22)';ctx.fillRect(x,y+h-bv*.4,w,bv*.4);ctx.fillRect(x+w-bv*.4,y,bv*.4,h);
    ctx.strokeStyle='rgba(30,20,10,.35)';ctx.lineWidth=2;ctx.strokeRect(x+bv,y+bv,w-bv*2,h-bv*2);
    b.fillStyle='#5a5a5a';b.fillRect(x,y,w,h);b.fillStyle='#707070';b.fillRect(x+bv,y+bv,w-bv*2,h-bv*2);
  }
  // carved ornament hint in the top panels
  ctx.strokeStyle='rgba(30,20,10,.32)';ctx.lineWidth=3;
  for(const cx of [.24,.56]){ctx.beginPath();ctx.ellipse(cx*ppm,.35*ppm,.07*ppm,.13*ppm,0,0,7);ctx.stroke();
    ctx.beginPath();ctx.moveTo((cx-.07)*ppm,.35*ppm);ctx.lineTo((cx+.07)*ppm,.35*ppm);ctx.stroke()}
  return {map:cm,bump:cb};
}

function outsideTex(){
  const W=512,H=512,c=canvas(W,H),x=c.getContext('2d'),r=mulberry32(9);
  const g=x.createLinearGradient(0,0,0,H);g.addColorStop(0,'#f6f7f1');g.addColorStop(.55,'#eef1e6');g.addColorStop(1,'#d9e2cc');
  x.fillStyle=g;x.fillRect(0,0,W,H);
  x.filter='blur(16px)';
  for(let i=0;i<70;i++){const px=r()*W,py=H*.25+r()*H*.8,rad=20+r()*70;
    x.fillStyle=`rgba(${110+r()*60},${140+r()*50},${90+r()*40},${.18+r()*.3})`;x.beginPath();x.arc(px,py,rad,0,7);x.fill()}
  x.filter='none';return c;
}

function paperTex(){
  const W=256,H=256,c=canvas(W,H),ctx=c.getContext('2d'),im=ctx.createImageData(W,H),nz=makeNoise(3);
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){const v=212+(nz.fbm(x*.08,y*.08,3)-.5)*14+(nz.n(x*.9,y*.9)-.5)*6;const i=(y*W+x)*4;
    im.data[i]=v;im.data[i+1]=v-1;im.data[i+2]=v-5;im.data[i+3]=255}
  ctx.putImageData(im,0,0);return c;
}

function dotSprite(){const c=canvas(64,64),x=c.getContext('2d'),g=x.createRadialGradient(32,32,0,32,32,32);
  g.addColorStop(0,'rgba(255,248,230,1)');g.addColorStop(.35,'rgba(255,240,210,.5)');g.addColorStop(1,'rgba(255,240,210,0)');
  x.fillStyle=g;x.fillRect(0,0,64,64);return c}

/* ---------- outside: birches, spruce line, sky ---------- */
function birchTex(seed){
  const W=256,H=1024,c=canvas(W,H),x=c.getContext('2d'),r=mulberry32(seed);
  const cx=W/2+(r()-.5)*20,lean=(r()-.5)*30;
  const tx=t=>cx+lean*t+Math.sin(t*5+seed)*4;            // trunk centre at height fraction t (0 = bottom)
  // drooping branches first, behind the trunk
  x.lineCap='round';
  for(let i=0;i<46;i++){
    const t=.32+r()*.66,y0=H*(1-t),sx=tx(t),dir=r()<.5?-1:1,len=40+r()*90;
    const ex=sx+dir*len,ey=y0-10-r()*30,hx=ex+dir*(10+r()*20),hy=ey+60+r()*160;
    x.strokeStyle=`rgba(${60+r()*30},${52+r()*20},${46+r()*20},${.75+r()*.25})`;x.lineWidth=.8+r()*1.6;
    x.beginPath();x.moveTo(sx,y0);x.quadraticCurveTo((sx+ex)/2,ey-10,ex,ey);x.quadraticCurveTo(hx,ey+10,hx,hy);x.stroke();
    // leaves along the hanging twig
    for(let k=0;k<26;k++){const u=r(),lx=ex+(hx-ex)*u+(r()-.5)*14,ly=ey+(hy-ey)*u+(r()-.5)*10;
      const yel=r();x.fillStyle=yel<.55?`rgb(${200+r()*40},${168+r()*40},${60+r()*30})`:yel<.85?`rgb(${150+r()*40},${158+r()*30},${70+r()*30})`:`rgb(${120+r()*20},${130+r()*20},${70+r()*20})`;
      x.beginPath();x.ellipse(lx,ly,2+r()*2.4,1.4+r()*1.6,r()*3,0,7);x.fill()}
  }
  // trunk
  for(let y=0;y<H;y+=2){const t=1-y/H,w=Math.max(3,22*(1-t*.82));
    x.fillStyle=`rgb(${226+r()*14},${224+r()*12},${214+r()*12})`;x.fillRect(tx(t)-w/2,y,w,2)}
  for(let i=0;i<180;i++){const t=r(),w=Math.max(3,22*(1-t*.82)),y=H*(1-t),xx=tx(t)-w/2+r()*w*.6;
    x.fillStyle=`rgba(20,20,18,${.55+r()*.4})`;x.fillRect(xx,y,2+r()*w*.5,1+r()*3)}
  const g=x.createLinearGradient(0,H*.86,0,H);g.addColorStop(0,'rgba(40,36,30,0)');g.addColorStop(1,'rgba(40,36,30,.85)');
  x.globalCompositeOperation='source-atop';x.fillStyle=g;x.fillRect(0,H*.86,W,H*.14);x.globalCompositeOperation='source-over';
  return c;
}
function spruceLineTex(){
  const W=2048,H=384,c=canvas(W,H),x=c.getContext('2d'),r=mulberry32(61);
  for(let layer=0;layer<2;layer++){
    for(let i=0;i<70;i++){const px=r()*W,h=H*(.45+r()*.5)*(layer?.8:1),w=h*(.28+r()*.1),base=H;
      x.fillStyle=layer?`rgb(${70+r()*10},${84+r()*10},${78+r()*8})`:`rgb(${92+r()*10},${104+r()*10},${98+r()*8})`;
      x.beginPath();x.moveTo(px,base-h);
      for(let k=1;k<=7;k++){const yy=base-h+h*k/7,ww=w*k/7;x.lineTo(px+ww*.6,yy-h*.03);x.lineTo(px+ww*.45,yy-h*.05)}
      x.lineTo(px+w*.5,base);x.lineTo(px-w*.5,base);
      for(let k=7;k>=1;k--){const yy=base-h+h*k/7,ww=w*k/7;x.lineTo(px-ww*.45,yy-h*.05);x.lineTo(px-ww*.6,yy-h*.03)}
      x.closePath();x.fill()}}
  return c;
}
function skyTex(){const c=canvas(32,512),x=c.getContext('2d'),g=x.createLinearGradient(0,0,0,512);
  g.addColorStop(0,'#c4c8c6');g.addColorStop(.6,'#dfe1dc');g.addColorStop(1,'#e9eae4');x.fillStyle=g;x.fillRect(0,0,32,512);return c}

const treeVS=`uniform float uTime,uWind,uPhase,uAmp;varying vec2 vUv;
void main(){vUv=uv;vec3 p=position;float h=uv.y;
float s=sin(uTime*1.25+uPhase+p.y*.21)+.55*sin(uTime*2.7+uPhase*1.7+p.x*.5)+.25*sin(uTime*6.1+uPhase*3.1+p.y*2.);
float bend=h*h*uAmp*uWind;p.x+=s*bend;p.z+=.35*s*bend;
gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.);}`;
const treeFS=`uniform sampler2D map;uniform vec3 tint,haze;uniform float hazeAmt,bright;varying vec2 vUv;
void main(){vec4 c=texture2D(map,vUv);if(c.a<.4)discard;vec3 col=mix(c.rgb*tint,haze,hazeAmt)*bright;gl_FragColor=vec4(col,1.);
#include <tonemapping_fragment>
#include <colorspace_fragment>
}`;

/* ---------- scene ---------- */
const params=new URLSearchParams(location.search);
const STILL=params.has('still');
const canvasEl=document.getElementById('c');
const loadmsg=document.getElementById('loadmsg');
/* the loading plate: the bar follows what has really arrived — every texture and model by its size in bytes,
   the room being built, the first weather film byte by byte, the shaders — and glides, never jumps */
const LOAD={man:null,doneB:0,totB:0,stages:0,vid:0,vidN:null,cmp:0,shown:0,item:'',fin:false,last:performance.now()};
fetch('manifest.json').then(r=>r.ok?r.json():null).then(m=>{LOAD.man=m}).catch(()=>{});
{ const LN={floor:'пол',plaster:'стены',red:'красная стена',door:'дверь',chair:'стул',stone:'камень',bulb:'лампочка',win:'окно',window:'окно',paper:'бумага',ceil:'потолок',calm:'ясно',wind:'ветер',shade:'пасмурно',rain:'дождь',storm:'гроза',detail:'вид из окна'},
    LK={diff:'цвет',nor:'рельеф',rough:'шероховатость',ao:'затенение',disp:'высота',alpha:'прозрачность',emis:'свечение',metal:'металл',sdf:'форма',hf:'форма'};
  const rel=u=>decodeURI(String(u)).replace(location.origin,'').replace(/^.*?\/(tex4k|tex)\//,'$1/').replace(/^\.?\//,'');
  const size=u=>{const m=LOAD.man,k=rel(u);return m&&m[k]?m[k]:260000};
  const mb=b=>(b/1048576).toFixed(b<1048576?2:1).replace('.',',')+' МБ';
  const name=u=>{const k=rel(u),f=k.split('/').pop(),b=f.replace(/\.[a-z0-9]+$/,''),[a,c]=b.split('_');let t=(LN[a]||a)+(c?' · '+(LK[c]||c):f.endsWith('.obj')?' · модель':'');if(k.startsWith('tex4k/'))t+=' 4K';return t};
  LOAD.name=name;LOAD.mb=mb;
  const DM=THREE.DefaultLoadingManager,is=DM.itemStart.bind(DM),ie=DM.itemEnd.bind(DM),ir=DM.itemError.bind(DM);
  DM.itemStart=u=>{LOAD.totB+=size(u);is(u)};
  DM.itemEnd=u=>{const b=size(u);LOAD.doneB+=b;LOAD.item=name(u)+' · '+mb(b);ie(u)};
  DM.itemError=u=>{LOAD.doneB+=size(u);ir(u)};
  window.__loadVid=(n,got,tot)=>{if(!LOAD.vidN)LOAD.vidN=n;if(n!==LOAD.vidN||!tot)return;LOAD.vid=got/tot;LOAD.item='погода за окном · '+mb(got)+' из '+mb(tot)};
  new MutationObserver(()=>{LOAD.stages++;loadmsg.classList.add('swap');requestAnimationFrame(()=>requestAnimationFrame(()=>loadmsg.classList.remove('swap')))}).observe(loadmsg,{childList:true,characterData:true,subtree:true});
  const bar=document.getElementById('loadbar'),num=document.getElementById('loadn'),item=document.getElementById('loaditem'),box=document.getElementById('loader');
  const tickL=()=>{const now=performance.now(),dt=Math.min(.1,(now-LOAD.last)/1000);LOAD.last=now;
    const fb=LOAD.totB?LOAD.doneB/LOAD.totB:0,target=LOAD.fin?1:Math.min(.985,.62*fb+.12*Math.min(1,LOAD.stages/7)+.18*LOAD.vid+.08*LOAD.cmp);
    LOAD.shown=Math.min(target,LOAD.shown+Math.max(.04*dt,(target-LOAD.shown)*Math.min(1,dt*4.5)));
    bar.style.transform=`scaleX(${LOAD.shown.toFixed(4)})`;num.textContent=Math.floor(LOAD.shown*100)+'%';if(LOAD.item&&item.textContent!==LOAD.item)item.textContent=LOAD.item;
    if(!box.classList.contains('done'))requestAnimationFrame(tickL)};
  requestAnimationFrame(tickL)}
let ENTRY=null;
let renderer;
try{renderer=new THREE.WebGLRenderer({canvas:canvasEl,antialias:true,powerPreference:'high-performance',preserveDrawingBuffer:STILL})}
catch(e){loadmsg.className='err';loadmsg.textContent='Этот браузер не смог запустить WebGL. Откройте страницу в свежем Chrome, Safari или Firefox.';throw e}
renderer.setPixelRatio(STILL?1:Math.min(window.devicePixelRatio||1,1.5));
renderer.shadowMap.enabled=true;renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.AgXToneMapping;renderer.toneMappingExposure=1.38;
renderer.outputColorSpace=THREE.SRGBColorSpace;
const maxAniso=renderer.capabilities.getMaxAnisotropy();
/* texture tier: computers with a real GPU get the full 4K maps (tex4k/), phones and tablets keep the lighter set; ?hq / ?lq force it */
const HQ=params.has('hq')||(!params.has('lq')&&Math.min(screen.width,screen.height)>=700&&!/iP(hone|ad|od)|Android|Mobile/i.test(navigator.userAgent)&&!(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1)&&renderer.capabilities.maxTextureSize>=8192&&(navigator.deviceMemory||8)>=4);
const TEX4K=new Set(['floor_diff.jpg','floor_nor.jpg','plaster_diff.jpg','plaster_nor.jpg','red_diff.jpg','red_nor.jpg','door_diff.jpg','door_nor.jpg','chair_diff.jpg','chair_nor.jpg','stone_diff.jpg','stone_nor.jpg']);
window.__HQ=HQ;
RectAreaLightUniformsLib.init();
const glassEnv=new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(),.04).texture;

const scene=new THREE.Scene();scene.background=new THREE.Color('#171612');window.__scene=scene;
const camera=new THREE.PerspectiveCamera(50,1,.05,120);   // driven by the controls
const view=new THREE.PerspectiveCamera(50,1,.05,120);     // what we render: camera + handheld drift + shake
const RW=3.6,RD=5.2,RH=3.2,X0=-RW/2,X1=RW/2,Z0=-RD/2,Z1=RD/2;
const WIN={z0:-2.45,z1:-1.05,y0:.92,y1:2.32,depth:.36};
const WC=new THREE.Vector3(X1,(WIN.y0+WIN.y1)/2,(WIN.z0+WIN.z1)/2);

function tex(c,{srgb=false,rep=[1,1]}={}){const t=new THREE.CanvasTexture(c);if(srgb)t.colorSpace=THREE.SRGBColorSpace;t.anisotropy=maxAniso;t.wrapS=t.wrapT=THREE.RepeatWrapping;t.repeat.set(rep[0],rep[1]);return t}
const shadowOnly=new THREE.MeshBasicMaterial({colorWrite:false,depthWrite:false});
function blocker(w,h,d,x,y,z){const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),shadowOnly);m.position.set(x,y,z);m.castShadow=true;scene.add(m);return m}

/* simulation state */
const S={trClosed:false,latched:false,broken:false,breakPending:false,storyT:-1,time:0,windOn:true,gust:0,interior:0,exterior:0,shake:0,taped:false,sunny:false,shot:3};
const gustNoise=t=>.55+.25*Math.sin(t*.9)+.15*Math.sin(t*2.3+1.3)+.12*Math.sin(t*5.1+.4);
const simRand=mulberry32(2024);

async function build(){
  const wallPal={a:[164,172,156],b:[146,160,158],c:[182,176,152],peel:[200,202,192],peelAt:.7,dirt:true};
  const ceilPal={a:[196,195,188],b:[176,176,170],c:[158,157,149],peel:[214,213,206],peelAt:.68,dirt:false};
  loadmsg.textContent='Загружаю фактуры…';await tick();
  const TL=new THREE.TextureLoader();
  const ld=(name,srgb)=>new Promise(res=>TL.load((HQ&&TEX4K.has(name)?'tex4k/':'tex/')+name,t=>{t.wrapS=t.wrapT=THREE.RepeatWrapping;t.anisotropy=maxAniso;if(srgb)t.colorSpace=THREE.SRGBColorSpace;res(t)},undefined,()=>res(null)));
  const TXN=['floor','plaster','red','door'],TX={};
  await Promise.all(TXN.flatMap(n=>[ld(n+'_diff.jpg',true).then(t=>TX[n+'D']=t),ld(n+'_nor.jpg').then(t=>TX[n+'N']=t),ld(n+'_rough.jpg').then(t=>TX[n+'R']=t)]).concat([ld('paper_nor.jpg').then(t=>TX.paperN=t),ld('floor_disp.png').then(t=>TX.floorH=t),ld('floor_ao.jpg').then(t=>TX.floorA=t),ld('plaster_ao.jpg').then(t=>TX.plasterA=t),ld('red_ao.jpg').then(t=>TX.redA=t),ld('ceil_diff.jpg',true).then(t=>TX.ceilD=t)]));
  await Promise.all([ld('win_diff.jpg',true).then(t=>TX.winD=t),ld('win_nor.jpg').then(t=>TX.winN=t),ld('win_rough.jpg').then(t=>TX.winR=t),ld('chair_diff.jpg',true).then(t=>TX.chD=t),ld('chair_nor.jpg').then(t=>TX.chN=t),ld('chair_rough.jpg').then(t=>TX.chR=t),ld('chair_sdf.png').then(t=>TX.sdf=t),
    new Promise(res=>new OBJLoader().load('tex/chair.obj',o=>{TX.chairObj=o;res()},undefined,()=>res())),
    new Promise(res=>new OBJLoader().load('tex/stone.obj',o=>{TX.stoneObj=o;res()},undefined,()=>res())),
    new Promise(res=>new OBJLoader().load('tex/bulb.obj',o=>{TX.bulbObj=o;res()},undefined,()=>res())),
    ld('stone_diff.jpg',true).then(t=>TX.stD=t),ld('stone_nor.jpg').then(t=>TX.stN=t),ld('stone_rough.jpg').then(t=>TX.stR=t),
    ld('bulb_diff.jpg',true).then(t=>TX.buD=t),ld('bulb_nor.jpg').then(t=>TX.buN=t),ld('bulb_rough.jpg').then(t=>TX.buR=t),ld('bulb_metal.jpg').then(t=>TX.buM=t),ld('bulb_emis.jpg',true).then(t=>TX.buE=t),ld('bulb_alpha.jpg').then(t=>TX.buA=t),
    fetch('tex/chair_hf.json').then(r=>r.ok?r.json():null).then(j=>{TX.hf=j}).catch(()=>{}),
    new Promise(res=>new OBJLoader().load('tex/window.obj',o=>{TX.winObj=o;res()},undefined,()=>res()))]);
  const hasT=n=>TX[n+'D']&&TX[n+'N']&&TX[n+'R'];
  const pbr=(n,rep,extra={})=>{const c=k=>{const t=TX[n+k].clone();t.needsUpdate=true;t.repeat.set(rep[0],rep[1]);if(extra.offset)t.offset.set(...extra.offset);return t};
    const m=new THREE.MeshStandardMaterial({map:c('D'),normalMap:c('N'),roughnessMap:c('R'),roughness:1,metalness:0,color:extra.color||'#ffffff',normalScale:new THREE.Vector2(extra.ns||1,extra.ns||1)});
    if(TX[n+'A']){m.aoMap=c('A');m.aoMapIntensity=extra.ao||1}return m};
  function crumpleNormal(W,H,seed,nBig,nSmall,amp,nz=.09){const c=canvas(W,H),x=c.getContext('2d'),img=x.createImageData(W,H),d=img.data,r=mulberry32(seed);
    const mk=n=>{const P=[];for(let i=0;i<n;i++)P.push([r()*W,r()*H,(r()-.5)*2,(r()-.5)*2]);return P};const B=mk(nBig),S=mk(nSmall);
    const cell=(P,X,Y)=>{let b=1e18,b2=1e18,k=0;for(let i=0;i<P.length;i++){let dx=X-P[i][0],dy=Y-P[i][1];if(dx>W/2)dx-=W;if(dx<-W/2)dx+=W;if(dy>H/2)dy-=H;if(dy<-H/2)dy+=H;const q=dx*dx+dy*dy;if(q<b){b2=b;b=q;k=i}else if(q<b2)b2=q}return [k,Math.sqrt(b2)-Math.sqrt(b)]};
    for(let Y=0;Y<H;Y++)for(let X=0;X<W;X++){const [kb,eb]=cell(B,X,Y),[ks,es]=cell(S,X,Y);
      let nx=B[kb][2]*.55+S[ks][2]*.3+(r()-.5)*nz,ny=B[kb][3]*.55+S[ks][3]*.3+(r()-.5)*nz;
      // a sharp little ridge along the crease lines
      if(eb<2.2){const s=1-eb/2.2;nx*=1+s;ny*=1+s}
      nx*=amp;ny*=amp;const l=Math.hypot(nx,ny,1),o=(Y*W+X)*4;d[o]=(nx/l*.5+.5)*255;d[o+1]=(ny/l*.5+.5)*255;d[o+2]=(1/l*.5+.5)*255;d[o+3]=255}
    x.putImageData(img,0,0);const t=new THREE.CanvasTexture(c);t.colorSpace=THREE.NoColorSpace;t.anisotropy=8;return t}
  const paperNorm=m=>{if(TX.paperN){m.normalMap=TX.paperN;m.normalScale=new THREE.Vector2(.8,.8);m.needsUpdate=true}m.roughness=.92;return m};
  loadmsg.textContent='Штукатурю стены…';await tick();
  const PL=hasT('plaster'),tLeft=PL?null:plasterTex(RD,RH,11,wallPal),tRight=PL?null:plasterTex(RD,RH,12,wallPal),tBack=PL?null:plasterTex(RW,RH,13,wallPal),tFront=hasT('red')?null:plasterTex(RW,RH,14,wallPal,120,512);
  loadmsg.textContent='Белю потолок…';await tick();
  const tCeil=plasterTex(RW,RD,21,ceilPal,160,768),tBeam=plasterTex(4,.6,22,{...ceilPal,peelAt:.64},200,1024),tReveal=PL?null:plasterTex(1.6,1.6,15,wallPal,200,384);
  loadmsg.textContent='Стелю половицы…';await tick();
  const tFloor=hasT('floor')?null:floorTex();
  loadmsg.textContent='Сажаю берёзы за окном…';await tick();
  const tDoor=doorTex(TX.doorD&&TX.doorD.image),tPaper=paperTex(),tBirch=[birchTex(3),birchTex(8),birchTex(15)],tSpruce=spruceLineTex(),tSky=skyTex();

  const plaster=(t,rep)=>new THREE.MeshStandardMaterial({map:tex(t.map,{srgb:true,rep}),bumpMap:tex(t.bump,{rep}),bumpScale:.6,roughness:.95});

  /* room shell */
  // soft contact occlusion on the floor under things that stand or lie on it (legs, stone, roll, door edge)
  const OCC={uOcc:{value:Array.from({length:10},()=>new THREE.Vector4(0,0,1,0))},uCapA:{value:[new THREE.Vector4(),new THREE.Vector4()]},uCapB:{value:[new THREE.Vector4(),new THREE.Vector4()]}};
  const grime=(m,kind)=>{m.onBeforeCompile=sh=>{if(kind==='floor')Object.assign(sh.uniforms,OCC);
    sh.vertexShader=sh.vertexShader.replace('#include <common>','#include <common>\nvarying vec3 vGW;').replace('#include <worldpos_vertex>','#include <worldpos_vertex>\nvGW=(modelMatrix*vec4(transformed,1.0)).xyz;');
    sh.fragmentShader=sh.fragmentShader.replace('#include <common>','#include <common>\nvarying vec3 vGW;'+(kind==='floor'?'\nuniform vec4 uOcc[10];uniform vec4 uCapA[2],uCapB[2];':'')).replace('#include <map_fragment>',`#include <map_fragment>
      float gdx=min(abs(vGW.x-(${X0.toFixed(3)})),abs(vGW.x-(${X1.toFixed(3)}))),gdz=min(abs(vGW.z-(${Z0.toFixed(3)})),abs(vGW.z-(${Z1.toFixed(3)})));
      float gn=fract(sin(dot(floor(vGW.xz*14.0+vGW.y*9.0),vec2(12.9898,78.233)))*43758.5453);
      diffuseColor.rgb*=.72+.28*exp(-distance(vGW,vec3(${WC.x.toFixed(2)},${WC.y.toFixed(2)},${WC.z.toFixed(2)}))/2.6);
      ${kind==='wall'||kind==='back'?`float gc=max(gdx,gdz);diffuseColor.rgb*=(1.0-${kind==='back'?'.12':'.3'}*(1.0-smoothstep(0.0,.45,vGW.y)))*(1.0-.14*smoothstep(${(RH-.5).toFixed(2)},${RH.toFixed(2)},vGW.y))*(1.0-.22*(1.0-smoothstep(0.0,.3,gc)))*(.97+.06*gn);`+
        (kind==='back'?`totalEmissiveRadiance+=diffuseColor.rgb*(.07+.11*(1.0-smoothstep(0.1,1.1,vGW.y)))*(.55+.45*exp(-distance(vGW.xz,vec2(${WC.x.toFixed(2)},${WC.z.toFixed(2)}))/2.4));`:''):
        kind==='ceil'?`float gc=min(gdx,gdz);diffuseColor.rgb*=(1.0-.25*(1.0-smoothstep(0.0,.5,gc)))*(.97+.05*gn);`:
        `float gc=min(gdx,gdz);diffuseColor.rgb*=(1.0-.28*(1.0-smoothstep(0.0,.35,gc)));
        float occ=1.0;for(int i=0;i<10;i++){vec4 o=uOcc[i];vec2 d=vGW.xz-o.xy;occ*=1.0-o.w*exp(-dot(d,d)/(o.z*o.z));}
        for(int i=0;i<2;i++){vec2 a=uCapA[i].xy,b=uCapA[i].zw,ab=b-a;float h=clamp(dot(vGW.xz-a,ab)/max(dot(ab,ab),1e-6),0.0,1.0);vec2 d=vGW.xz-a-ab*h;occ*=1.0-uCapB[i].y*exp(-dot(d,d)/(uCapB[i].x*uCapB[i].x));}
        diffuseColor.rgb*=occ;`}`)};m.customProgramCacheKey=()=>'grime2-'+kind;return m};
  const fm=hasT('floor')?pbr('floor',[RW/1.6,RD/1.6],{ns:2.3,color:'#f2ece6',ao:1.3}):new THREE.MeshStandardMaterial({map:tex(tFloor.map,{srgb:true}),roughnessMap:tex(tFloor.rough),bumpMap:tex(tFloor.bump),bumpScale:1.4,roughness:1,metalness:0});
  const floorTexW=RW;fm.map.repeat.set(1,1);
  let floorH=null;const FS=.016,FTOP=.83;
  const lowEnd=Math.min(screen.width,screen.height)<700&&!STILL;
  let floorGeo=new THREE.PlaneGeometry(RW,RD);
  if(hasT('floor')&&TX.floorH){
    const im=TX.floorH.image,fc=canvas(im.width,im.height),fx=fc.getContext('2d');fx.drawImage(im,0,0);
    const dd=fx.getImageData(0,0,fc.width,fc.height).data,HW=fc.width,HH=fc.height,hf=new Float32Array(HW*HH);
    for(let i=0;i<HW*HH;i++)hf[i]=FS*(dd[i*4]/255-FTOP);
    const g=(i,j)=>hf[((j%HH+HH)%HH)*HW+((i%HW+HW)%HW)];
    floorH=(x,z)=>{let u=((x-X0)/1.6)%1;if(u<0)u+=1;let v=((RD-(z-Z0))/1.6)%1;if(v<0)v+=1;
      const px=u*HW-.5,py=(1-v)*HH-.5,x0=Math.floor(px),y0=Math.floor(py),tx=px-x0,ty=py-y0;
      return (g(x0,y0)*(1-tx)+g(x0+1,y0)*tx)*(1-ty)+(g(x0,y0+1)*(1-tx)+g(x0+1,y0+1)*tx)*ty};
    floorGeo=lowEnd?new THREE.PlaneGeometry(RW,RD,200,290):new THREE.PlaneGeometry(RW,RD,380,550);
    const dm2=TX.floorH.clone();dm2.needsUpdate=true;dm2.repeat.set(RW/1.6,RD/1.6);
    fm.displacementMap=dm2;fm.displacementScale=FS;fm.displacementBias=-FS*FTOP;fm.needsUpdate=true}
  grime(fm,'floor');const floor=new THREE.Mesh(floorGeo,fm);floor.rotation.x=-Math.PI/2;floor.receiveShadow=true;scene.add(floor);
  // ceiling: old whitewash over concrete, cracked and flaking (as in the reference room)
  const ceilMat=(rep,col)=>{if(!hasT('red')||!TX.ceilD)return hasT('plaster')?pbr('plaster',rep,{color:col}):plaster(tCeil);const m=pbr('red',rep,{color:col,ns:1.5,ao:1.1});const t=TX.ceilD.clone();t.needsUpdate=true;t.repeat.set(rep[0],rep[1]);m.map=t;return m};
  const ceil=new THREE.Mesh(new THREE.PlaneGeometry(RW,RD),ceilMat([RW/2.1,RD/2.1],'#cdcdc6'));grime(ceil.material,'ceil');ceil.rotation.x=Math.PI/2;ceil.position.y=RH;ceil.receiveShadow=true;scene.add(ceil);
  const left=new THREE.Mesh(new THREE.PlaneGeometry(RD,RH+.08),(hasT('plaster')?pbr('plaster',[RD/2.9,RH/2.9],{color:'#f2f5ec',ns:1.9,ao:1.25}):plaster(tLeft)));grime(left.material,'wall');left.rotation.y=Math.PI/2;left.position.set(X0,(RH-.08)/2,0);left.receiveShadow=true;scene.add(left);
  const back=new THREE.Mesh(new THREE.PlaneGeometry(RW,RH+.08),(hasT('plaster')?pbr('plaster',[RW/2.9,RH/2.9],{color:'#f2f5ec',offset:[.37,0],ns:1.9,ao:1.25}):plaster(tBack)));grime(back.material,'back');back.position.set(0,(RH-.08)/2,Z0);back.receiveShadow=true;scene.add(back);
  const front=new THREE.Mesh(new THREE.PlaneGeometry(RW,RH+.08),(hasT('red')?pbr('red',[RW/1.8,RH/1.8],{color:'#ddd5cf',ns:1.7,ao:1.15}):plaster(tFront)));grime(front.material,'wall');front.rotation.y=Math.PI;front.position.set(0,(RH-.08)/2,Z1);front.receiveShadow=true;scene.add(front);
  const sh=new THREE.Shape();sh.moveTo(Z0,-.08);sh.lineTo(Z1,-.08);sh.lineTo(Z1,RH);sh.lineTo(Z0,RH);sh.lineTo(Z0,-.08);
  const hole=new THREE.Path();hole.moveTo(WIN.z0,WIN.y0);hole.lineTo(WIN.z0,WIN.y1);hole.lineTo(WIN.z1,WIN.y1);hole.lineTo(WIN.z1,WIN.y0);hole.lineTo(WIN.z0,WIN.y0);sh.holes.push(hole);
  const rmat=hasT('plaster')?pbr('plaster',[1/2.9,1/2.9],{color:'#f2f5ec',offset:[.61,0],ns:1.9,ao:1.25}):plaster(tRight,[1/RD,1/RH]);if(!hasT('plaster')){rmat.map.offset.set(.5,0);rmat.bumpMap.offset.set(.5,0)}
  const right=new THREE.Mesh(new THREE.ShapeGeometry(sh),rmat);grime(right.material,'wall');right.rotation.y=-Math.PI/2;right.position.x=X1;right.receiveShadow=true;scene.add(right);
  const rv=hasT('plaster')?pbr('plaster',[.3,.3],{color:'#f2f5ec',ns:1.9,ao:1.25}):plaster(tReveal);const D=WIN.depth,xr=X1+.004+D/2;
  const rbox=(w,h,d,x,y,z)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),rv);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;scene.add(m)};
  const GAP=-.01;
  rbox(D,WIN.y0-.015+.4,RD+1,xr,(WIN.y0-.015-.4)/2,0);   // under the frame's bottom rail, hidden by the sill
  rbox(D,RH+.6-WIN.y1-GAP,RD+1,xr,(WIN.y1+GAP+RH+.6)/2,0);
  rbox(D,WIN.y1-WIN.y0+2*GAP,WIN.z0-GAP-(Z0-.5),xr,(WIN.y0+WIN.y1)/2,(WIN.z0-GAP+Z0-.5)/2);
  rbox(D,WIN.y1-WIN.y0+2*GAP,Z1+.5-WIN.z1-GAP,xr,(WIN.y0+WIN.y1)/2,(WIN.z1+GAP+Z1+.5)/2);
  blocker(RW+1.4,.3,RD+1,0,RH+.16,0);blocker(RW+1.4,RH+.6,.3,0,RH/2,Z0-.16);blocker(.3,RH+.6,RD+1,X0-.16,RH/2,0);blocker(RW+1.4,RH+.6,.3,0,RH/2,Z1+.16);
  // skirting boards along the walls
  const skM=new THREE.MeshStandardMaterial({color:'#5b5246',roughness:.62});if(TX.doorN){const n=TX.doorN.clone();n.needsUpdate=true;n.repeat.set(3,.15);skM.normalMap=n;skM.normalScale=new THREE.Vector2(.6,.6)}
  const sk=(w,d,x,z,ry)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,.085,d),skM);m.position.set(x,.0425,z);m.rotation.y=ry;m.castShadow=true;m.receiveShadow=true;scene.add(m)};
  const bm=ceilMat([1.6,.3],'#c8c8c1');
  for(const z of [Z0+.13,-.55,1.35]){const b=new THREE.Mesh(new THREE.BoxGeometry(RW,.24,.26),bm);b.position.set(0,RH-.12,z);b.castShadow=true;b.receiveShadow=true;scene.add(b)}

  /* window: lining, sill, two casements that open inward */
  const wood=new THREE.MeshStandardMaterial({color:'#e6e3da',roughness:.72});
  // old window glass: almost clear, a faint green, reflections of the room; dust thicker toward the bottom and the edges, dried rain spots, a few smears
  function glassDirt(){const W=512,c=canvas(W,W),x=c.getContext('2d'),r=mulberry32(909),rc=canvas(W,W),y=rc.getContext('2d');
    x.fillStyle='rgba(232,238,235,.07)';x.fillRect(0,0,W,W);y.fillStyle='rgb(10,10,10)';y.fillRect(0,0,W,W);
    const blob=(px,py,rad,a,col,rough)=>{const g=x.createRadialGradient(px,py,0,px,py,rad);g.addColorStop(0,`rgba(${col},${a})`);g.addColorStop(1,`rgba(${col},0)`);x.fillStyle=g;x.fillRect(px-rad,py-rad,rad*2,rad*2);
      const g2=y.createRadialGradient(px,py,0,px,py,rad);g2.addColorStop(0,`rgba(${rough},${rough},${rough},${a})`);g2.addColorStop(1,'rgba(0,0,0,0)');y.fillStyle=g2;y.fillRect(px-rad,py-rad,rad*2,rad*2)};
    for(let i=0;i<380;i++){const px=r()*W,py=W*Math.pow(r(),.55),edge=Math.max(1-Math.min(px,W-px,py,W-py)/70,0);blob(px,py,8+r()*40,(.012+.12*edge*edge+.03*py/W)*r(),'190,184,172',150)}
    for(let i=0;i<180;i++){const px=r()*W,py=r()*W,rad=2+r()*7;x.strokeStyle=`rgba(170,165,152,${.15+r()*.25})`;x.lineWidth=.8+r();x.beginPath();x.arc(px,py,rad,0,6.3);x.stroke();
      y.strokeStyle='rgba(140,140,140,.4)';y.beginPath();y.arc(px,py,rad,0,6.3);y.stroke()}
    for(let i=0;i<26;i++){const px=r()*W,py=r()*W*.6,len=40+r()*160;x.strokeStyle=`rgba(185,180,168,${.08+r()*.12})`;x.lineWidth=1+r()*3;x.beginPath();x.moveTo(px,py);x.bezierCurveTo(px+(r()-.5)*8,py+len*.3,px+(r()-.5)*10,py+len*.7,px+(r()-.5)*6,py+len);x.stroke()}
    for(let i=0;i<3;i++){const px=W*(.75+r()*.2),py=W*(.35+r()*.3);for(let k=0;k<5;k++)blob(px+(r()-.5)*40,py+(r()-.5)*40,10+r()*14,.07,'205,200,190',110)}
    return [c,rc]}
  const [gdC,grC]=glassDirt(),gdT=tex(gdC,{srgb:true}),grTx=tex(grC);gdT.wrapS=gdT.wrapT=grTx.wrapS=grTx.wrapT=THREE.ClampToEdgeWrapping;
  const glass=new THREE.MeshPhysicalMaterial({color:'#f4f8f6',map:gdT,roughness:1,roughnessMap:grTx,metalness:0,ior:1.52,specularIntensity:1,transparent:true,opacity:.42,depthWrite:false,envMapIntensity:1.25});
  // glass reflects more the flatter you look at it (Fresnel): the room shows in the panes at an angle, they almost vanish face-on
  glass.onBeforeCompile=sh=>{sh.fragmentShader=sh.fragmentShader.replace('#include <opaque_fragment>',`{float fr=pow(1.0-abs(dot(normalize(normal),normalize(vViewPosition))),5.0);
      diffuseColor.a=clamp(diffuseColor.a+.035+fr*.6,0.,1.);}
    #include <opaque_fragment>`)};glass.customProgramCacheKey=()=>'glassF';
  const pm=paperNorm(new THREE.MeshStandardMaterial({map:tex(tPaper,{srgb:true}),roughness:.82,side:THREE.DoubleSide}));
  const tapeC=canvas(64,128),tx2=tapeC.getContext('2d'),tr2=mulberry32(4);
  tx2.fillStyle='#d9a868';tx2.fillRect(0,0,64,128);
  for(let k=0;k<160;k++){tx2.fillStyle=`rgba(${120+tr2()*60},${80+tr2()*40},${40+tr2()*30},${.05+tr2()*.12})`;tx2.fillRect(tr2()*64,tr2()*128,1,4+tr2()*20)}
  tx2.globalCompositeOperation='destination-out';
  for(const yy of [0,128]){tx2.beginPath();tx2.moveTo(0,yy);for(let x=0;x<=64;x+=4)tx2.lineTo(x,yy+(yy?-1:1)*(2+tr2()*7));tx2.lineTo(64,yy);tx2.fill()}
  tx2.globalCompositeOperation='source-over';
  const tm=new THREE.MeshStandardMaterial({map:tex(tapeC,{srgb:true}),alphaTest:.5,roughness:.55,side:THREE.DoubleSide});tm.map.wrapS=tm.map.wrapT=THREE.ClampToEdgeWrapping;
  const flapU={uTime:{value:0},uWind:{value:0},uWin:{value:WC.clone()}};
  { const cn=crumpleNormal(384,544,303,10,46,.075);pm.normalMap=cn;pm.normalScale=new THREE.Vector2(1,1);pm.needsUpdate=true}
  const pmRest=pm.clone();
  pm.onBeforeCompile=sh=>{Object.assign(sh.uniforms,flapU);
    sh.vertexShader='uniform float uTime,uWind;uniform vec3 uWin;\n'+sh.vertexShader.replace('#include <begin_vertex>',`#include <begin_vertex>
#ifdef USE_INSTANCING
  vec3 ipos=vec3(instanceMatrix[3][0],instanceMatrix[3][1],instanceMatrix[3][2]);
#else
  vec3 ipos=vec3(modelMatrix[3][0],modelMatrix[3][1],modelMatrix[3][2]);
#endif
  float tph=fract(sin(dot(ipos.xz+ipos.y,vec2(12.9898,78.233)))*43758.5453)*6.2831;
  float tt=clamp(-position.y/0.42,0.,1.),xn=position.x/0.1485,prox=exp(-distance(ipos,uWin)/1.4);
  transformed.z*=aCurlV;
  float amp=uWind*(0.003+0.03*prox)*aFreeV;
  float fq=2.4+2.8*fract(tph*1.7);
  transformed.z+=amp*tt*tt*(.5+.5*sin(uTime*(fq+2.0*prox)+tph+tt*3.5+xn*.9))+amp*.8*tt*xn*xn*(.5+.5*sin(uTime*(fq*1.9)+tph*1.7+xn*2.3));`).replace('#include <common>',`#include <common>
#ifdef USE_INSTANCING
  attribute float aFree,aCurl;
  #define aFreeV aFree
  #define aCurlV aCurl
#else
  #define aFreeV 1.0
  #define aCurlV 1.0
#endif`)};
  pm.customProgramCacheKey=()=>'paperflap3';
  const fbox=(g,w,h,d,x,y,z)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),wood);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;g.add(m);return m};
  /* the window: an old wooden double casement with a small vent pane (fortochka) in the right leaf */
  const wW=WIN.z1-WIN.z0,wH=WIN.y1-WIN.y0,SM=.01,XC=X1+.19,ZC=WC.z,GLX=XC-.148;   // the frame box stands a little proud of the wall, hinges at the wall face, so the casements can swing right round to the wall
  const W2M=new THREE.Matrix4().set(0,0,SM,XC, 0,SM,0,WIN.y0, SM,0,0,ZC, 0,0,0,1);   // model (cm) -> world, mirrored so the fortochka is on the right
  const lw=.629+.018,lh=wH-.06,tapeY=lh*.5;
  const hingeA=ZC-.629,hingeB=ZC+.629;
  const winMat=new THREE.MeshStandardMaterial({color:'#f4f2ec',roughness:1});
  if(TX.winD){winMat.map=TX.winD;winMat.normalMap=TX.winN;winMat.roughnessMap=TX.winR;winMat.normalScale=new THREE.Vector2(1.2,1.2)}else{winMat.roughness=.7}
  glass.side=THREE.DoubleSide;
  const parts={};
  if(TX.winObj)TX.winObj.traverse(o=>{if(!o.isMesh)return;const g=o.geometry,mats=Array.isArray(o.material)?o.material:[o.material];
    const groups=g.groups.length?g.groups:[{start:0,count:g.attributes.position.count,materialIndex:0}];
    for(const gr of groups){const name=(mats[gr.materialIndex]||mats[0]).name;const ng=new THREE.BufferGeometry();
      for(const k of ['position','normal','uv']){const a=g.attributes[k];if(!a)continue;ng.setAttribute(k,new THREE.BufferAttribute(a.array.slice(gr.start*a.itemSize,(gr.start+gr.count)*a.itemSize),a.itemSize))}
      (parts[o.name]=parts[o.name]||{})[name]=ng}});
  const placed=(geo,localFromWorld)=>{const g2=geo.clone();const M=new THREE.Matrix4().multiplyMatrices(localFromWorld,W2M);g2.applyMatrix4(M);
    if(M.determinant()<0){for(const k of ['position','normal','uv']){const a=g2.attributes[k];if(!a)continue;const it=a.itemSize,arr=a.array;
      for(let t=0;t<arr.length;t+=3*it)for(let c=0;c<it;c++){const x=arr[t+it+c];arr[t+it+c]=arr[t+2*it+c];arr[t+2*it+c]=x}}}
    g2.computeBoundingBox();return g2};
  const addPart=(geo,mat,parent,lfw)=>{const m=new THREE.Mesh(placed(geo,lfw),mat);m.castShadow=mat!==glass;m.receiveShadow=true;parent.add(m);return m};
  const paneOf=(m,parent)=>{const b=m.geometry.boundingBox;return {m,g:parent,cx:(b.min.x+b.max.x)/2,cy:(b.min.y+b.max.y)/2,cz:(b.min.z+b.max.z)/2,w:b.max.x-b.min.x,h:b.max.y-b.min.y}};
  const ALLPANES=[];
  let fr0=null;if(parts.win1){const fr=new THREE.Group();fr0=fr;scene.add(fr);addPart(parts.win1.window_1,winMat,fr,new THREE.Matrix4());
  }
  // a plain painted casing round the frame where it leaves the wall
  // the frame sits flush in the opening; a flat painted casing (nalichnik) laps 1.5 cm over it on three sides and stands on the sill, so the joint is never seen
  const CAS={w:.072,t:.02,ov:.015};
  { const cw=CAS.w,ct=CAS.t,ov=CAS.ov,cm=winMat.clone();cm.color.set('#e9e7e0');
    const cb=(w,hh,d,x,y,z)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(w,hh,d),cm);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;scene.add(m)};
    const y0=WIN.y0+.0035,y1=WIN.y1-ov+cw;
    cb(ct,y1-y0,cw,X1-ct/2,(y0+y1)/2,WIN.z0+ov-cw/2);cb(ct,y1-y0,cw,X1-ct/2,(y0+y1)/2,WIN.z1-ov+cw/2);
    cb(ct,cw,WIN.z1-WIN.z0-2*ov+2*cw,X1-ct/2-.0005,WIN.y1-ov+cw/2,(WIN.z0+WIN.z1)/2)}
  // interior sill board: painted wood, 6 cm proud of the wall, ears tucked into the plaster
  const SILL={x0:X1-.115,x1:X1+.014,top:WIN.y0+.0035,z0:WIN.z0-.075,z1:WIN.z1+.075};   // runs in under the frame's bottom rail; its top clears the plaster below the opening and stays under the casements
  { const t=.034,r=.009,sp=new THREE.Shape();
    sp.moveTo(SILL.x1,SILL.top-t);sp.lineTo(SILL.x0+r*.6,SILL.top-t);sp.quadraticCurveTo(SILL.x0,SILL.top-t,SILL.x0,SILL.top-t+r);
    sp.lineTo(SILL.x0,SILL.top-r);sp.quadraticCurveTo(SILL.x0,SILL.top,SILL.x0+r,SILL.top);sp.lineTo(SILL.x1,SILL.top);sp.lineTo(SILL.x1,SILL.top-t);
    const g=new THREE.ExtrudeGeometry(sp,{depth:SILL.z1-SILL.z0,bevelEnabled:false,curveSegments:6});g.translate(0,0,SILL.z0);
    const sm=winMat.clone();if(sm.map){for(const k of ['map','normalMap','roughnessMap']){const tx=sm[k].clone();tx.needsUpdate=true;tx.wrapS=tx.wrapT=THREE.RepeatWrapping;tx.repeat.set(1.6,1.6);sm[k]=tx}}
    sm.color.set('#ecebe5');const sill=new THREE.Mesh(g,sm);sill.castShadow=true;sill.receiveShadow=true;scene.add(sill)}
  function leaf(hingeZ,side,name){
    const g=new THREE.Group(),d=.09;
    g.position.set(XC-.14,WIN.y0,hingeZ);g.rotation.y=side>0?-Math.PI/2:Math.PI/2;g.updateMatrix();
    const lfw=g.matrix.clone().invert();const panes=[];
    if(parts[name]){addPart(parts[name].window_1,winMat,g,lfw);if(parts[name].glass){const gm=addPart(parts[name].glass,glass,g,lfw);const pn=paneOf(gm,g);panes.push(pn);ALLPANES.push(pn)}}
    const tp=new THREE.Group();tp.position.set(lw-.05,tapeY,side*(d/2+.003));g.add(tp);
    const tpm=new THREE.Mesh(new THREE.PlaneGeometry(.068,.05),tm);tpm.position.x=.034;tpm.rotation.z=side*.03;tp.add(tpm);
    scene.add(g);
    // hang it on its real hinge line: the room-side corner of its outer edge (so it swings clear of the frame box)
    g.updateMatrixWorld(true);const bb=new THREE.Box3();g.children.forEach(c=>{if(c.isMesh)bb.expandByObject(c)});
    const P=new THREE.Vector3(bb.min.x,WIN.y0,side>0?bb.min.z:bb.max.z),dl=g.position.clone().sub(P).applyQuaternion(g.quaternion.clone().invert());
    g.children.forEach(c=>c.position.add(dl));g.position.copy(P);g.updateMatrix();g.updateMatrixWorld(true);
    const lb=new THREE.Box3();g.children.forEach(c=>{if(c.isMesh){c.geometry.computeBoundingBox();const b=c.geometry.boundingBox.clone();b.min.add(c.position);b.max.add(c.position);lb.union(b)}});
    return {g,tp,side,hingeZ:P.z,hx:P.x,wb:bb.clone(),lb,th:0,om:0,max:0,rot:0,panes,lfw:g.matrix.clone().invert()};
  }
  const leafA=leaf(hingeA,1,'win3'),leafB=leaf(hingeB,-1,'win2');
  // how far each casement can swing before it meets the plaster reveal (or the papered wall)
  // (both faces of the sash, 5 cm thick, against the wall face, the casing, the reveal and the room's other walls)
  const maxOpenLine=(l,len,hz)=>{len=len||lw;hz=hz===undefined?l.hingeZ:hz;for(let th=.6;th<3.12;th+=.003){const dx=-Math.sin(th),dz=l.side*Math.cos(th),nx=dz,nz=-dx;
      for(let t=0;t<=len;t+=.01)for(const o of [-.026,.026]){const x=XC-.14+dx*t+nx*o,z=hz+dz*t+nz*o;
        const beyond=l.side>0?z<WIN.z0+CAS.ov+.002:z>WIN.z1-CAS.ov-.002;
        if(beyond&&x>X1-CAS.t-.003)return th-.003;
        if(z<Z0+.03||z>Z1-.03)return th-.003}}return 3.1};
  // the frame box: solid painted boards from each hinge line out to the plaster, over the head and under the sashes, so the closed window is one piece
  const BOX={x0:Math.min(leafA.hx,leafB.hx)+.0004,x1:X1+.13};
  { const bm=winMat.clone();if(bm.map){for(const k of ['map','normalMap','roughnessMap']){const tx=bm[k].clone();tx.needsUpdate=true;tx.wrapS=tx.wrapT=THREE.RepeatWrapping;tx.repeat.set(.6,2.2);bm[k]=tx}}bm.color.set('#eceae3');
    const bx=(x0,x1,y0,y1,z0,z1)=>{const m=new THREE.Mesh(new THREE.BoxGeometry(x1-x0,y1-y0,z1-z0),bm);m.position.set((x0+x1)/2,(y0+y1)/2,(z0+z1)/2);m.castShadow=true;m.receiveShadow=true;scene.add(m);return m};
    const top=Math.max(leafA.wb.max.y,leafB.wb.max.y)+.0012,bot=Math.min(leafA.wb.min.y,leafB.wb.min.y)-.0012;BOX.top=top;BOX.bot=bot;
    bx(BOX.x0,BOX.x1,SILL.top,WIN.y1,WIN.z0-.002,leafA.hingeZ-.0004);
    bx(BOX.x0,BOX.x1,SILL.top,WIN.y1,leafB.hingeZ+.0004,WIN.z1+.002);
    bx(BOX.x0,BOX.x1,top,WIN.y1,WIN.z0,WIN.z1);
    if(bot>SILL.top+.001)bx(BOX.x0,X1+.02,SILL.top,bot,WIN.z0,WIN.z1)}
  // how far a sash turns before any part of it meets the box, the casing, the wall or the room's side walls: its real outline, swept round the hinge
  function maxOpen(l){const pts=[],b=l.lb;for(let x=b.min.x;x<=b.max.x+1e-6;x+=.01)pts.push([x,b.min.z],[x,b.max.z]);for(let z=b.min.z;z<=b.max.z;z+=.01)pts.push([b.min.x,z],[b.max.x,z]);
    const q=new THREE.Quaternion(),v=new THREE.Vector3();
    const hit=(x,z)=>{if(z<Z0+.03||z>Z1-.03)return true;if(x>X1-.0008&&(z<WIN.z0||z>WIN.z1))return true;
      if(x>X1-CAS.t-.002&&((z<WIN.z0+CAS.ov&&z>WIN.z0+CAS.ov-CAS.w)||(z>WIN.z1-CAS.ov&&z<WIN.z1-CAS.ov+CAS.w)))return true;
      if(x>BOX.x0-.002&&(z<leafA.hingeZ-.0005||z>leafB.hingeZ+.0005))return true;return false};
    const Y1=new THREE.Vector3(0,1,0);for(let th=.04;th<3.4;th+=.004){q.setFromAxisAngle(Y1,l.rot(th));for(const [x,z] of pts){v.set(x,0,z).applyQuaternion(q).add(l.g.position);if(hit(v.x,v.z))return th-.004}}return 3.4}
  leafA.I=1;leafB.I=1.1;leafA.seed=1.3;leafB.seed=4.7;leafA.fric=.06;leafB.fric=.09;
  leafA.rot=th=>-Math.PI/2-th;leafB.rot=th=>Math.PI/2+th;
  leafA.max=maxOpen(leafA);leafB.max=maxOpen(leafB);
  const leaves=[leafA,leafB];
  // the fortochka: hinged on its outer edge inside the right casement, opens inward on its own
  const transom={g:new THREE.Group(),th:0,om:0,max:1.25,pane:null,absMax:maxOpenLine(leafB,.5,hingeB-.054)};
  { const hw=new THREE.Vector3(57.5,93,-12).applyMatrix4(W2M).applyMatrix4(leafB.lfw);
    transom.g.position.copy(hw);leafB.g.add(transom.g);transom.g.updateMatrix();
    const lfw=new THREE.Matrix4().multiplyMatrices(transom.g.matrix.clone().invert(),leafB.lfw);
    if(parts.win4){addPart(parts.win4.window_1,winMat,transom.g,lfw);if(parts.win4.glass){const gm=addPart(parts.win4.glass,glass,transom.g,lfw);transom.pane=paneOf(gm,transom.g);ALLPANES.push(transom.pane)}}}
  window.__frprof=()=>{const o={};fr0.updateMatrixWorld(true);fr0.traverse(m=>{if(!m.isMesh)return;const P=m.geometry.attributes.position,v=new THREE.Vector3();for(let i=0;i<P.count;i++){v.fromBufferAttribute(P,i).applyMatrix4(m.matrixWorld);if(v.y>1.0)continue;const k=(Math.round(v.x*100)/100).toFixed(2)+'|'+(Math.round(v.y*100)/100).toFixed(2);o[k]=(o[k]||0)+1}});return o};
  window.__leaves=()=>({A:[leafA.max,leafA.soft,leafA.th],B:[leafB.max,leafB.th],T:[transom.absMax,transom.th]});
  window.__try=(vh,ax,az)=>{setAim(ax,az);resetBodies();launch(vh);let touches=0,was=false,t1=-1,k=0;
    for(;k<5*480;k++){physStep(1/480,0);if(ST.contact&&!was){touches++;if(t1<0)t1=k}was=ST.contact;if(ST.hit)break}
    const r=[+vh.toFixed(2),ax,az,ST.hit?ST.leg:-1,+ST.hitV.toFixed(2),touches,+((k-t1)/480).toFixed(2),+STB.p.y.toFixed(3)];resetBodies();return r};
  window.__srch=vh=>{window.__rec={out:[]};resetBodies();launch(vh);for(let k=0;k<4*480;k++)physStep(1/480,0);const o=window.__rec.out;window.__rec=null;resetBodies();return o};
  window.__winbb=()=>{setLeaves(false);const out={};scene.updateMatrixWorld(true);const bb=o=>{const b=new THREE.Box3().setFromObject(o);return [b.min.toArray().map(v=>+v.toFixed(3)),b.max.toArray().map(v=>+v.toFixed(3))]};
    out.frame=bb(fr0);out.leafA=bb(leafA.g);out.leafB=bb(leafB.g);out.WIN=WIN;out.X1=X1;return out};
  function setLeaves(open){for(const l of leaves){l.th=open?(l.soft?Math.min(l.max*.96,l.soft+.01):l.max*.96):0;l.om=0;l.g.rotation.y=l.rot(l.th)}transom.th=open&&!S.trClosed?transom.max*.8:0;transom.om=0;transom.g.rotation.y=transom.th}

  /* outside: filmed weather over the rooftops, painted on a far dome round the window, one film dissolving into the next */
  const WXC={"calm":[[1.0,1.021,1.044,1.068,1.083,1.091,1.096,1.1,1.104,1.106,1.11,1.113,1.116,1.118,1.121,1.122,1.126,1.127,1.129,1.13,1.133,1.134,1.136,1.137,1.14,1.142,1.145,1.147,1.151,1.153,1.155,1.151,1.145,1.144,1.143,1.141,1.138,1.136,1.135,1.133,1.13,1.129,1.127,1.127,1.123,1.121,1.118,1.116,1.114,1.111,1.107,1.104,1.1,1.097,1.092,1.085,1.074,1.052,1.028,1.006],[1.0,0.984,1.008,1.079,1.163,1.233,1.277,1.293,1.298,1.297,1.298,1.299,1.291,1.272,1.266,1.257,1.212,1.157,1.159,1.198,1.25,1.284,1.302,1.302,1.305,1.301,1.286,1.208,1.064,0.937,0.908,1.032,1.17,1.269,1.301,1.303,1.304,1.303,1.292,1.261,1.215,1.167,1.152,1.198,1.249,1.263,1.268,1.281,1.297,1.296,1.296,1.295,1.297,1.286,1.249,1.185,1.1,1.02,0.983,0.994]],"wind":[[1.0,1.018,1.03,1.037,1.043,1.051,1.061,1.069,1.076,1.081,1.085,1.089,1.091,1.091,1.089,1.082,1.076,1.08,1.08,1.086,1.1,1.11,1.115,1.11,1.105,1.1,1.092,1.082,1.09,1.094,1.097,1.09,1.086,1.09,1.098,1.103,1.106,1.111,1.112,1.1,1.09,1.081,1.079,1.081,1.082,1.087,1.092,1.09,1.089,1.085,1.081,1.077,1.07,1.064,1.052,1.045,1.038,1.032,1.022,1.005],[1.0,1.001,1.004,1.003,1.003,1.002,1.003,1.0,1.001,0.999,1.0,0.999,0.923,0.966,0.981,0.867,0.678,0.602,0.534,0.556,0.724,0.939,0.959,0.7,0.573,0.555,0.616,0.777,0.583,0.572,0.575,0.575,0.726,0.656,0.564,0.558,0.649,0.942,0.962,0.775,0.588,0.531,0.581,0.662,0.834,0.955,0.979,0.914,0.992,0.997,1.0,1.0,1.001,1.002,1.003,1.002,1.004,1.004,1.005,1.003]],"shade":[[1.0,1.006,1.019,1.028,1.037,1.046,1.056,1.064,1.073,1.079,1.087,1.093,1.099,1.104,1.111,1.116,1.121,1.125,1.129,1.132,1.136,1.138,1.142,1.144,1.148,1.15,1.154,1.156,1.16,1.164,1.165,1.161,1.158,1.154,1.151,1.149,1.146,1.143,1.139,1.136,1.134,1.131,1.126,1.123,1.117,1.113,1.107,1.101,1.095,1.089,1.082,1.075,1.068,1.058,1.048,1.04,1.031,1.022,1.012,1.003],[1.0,1.003,1.006,1.007,1.009,1.01,1.012,1.012,1.014,1.014,1.017,1.016,1.02,1.02,1.023,1.023,1.024,1.025,1.029,1.029,1.031,1.016,1.009,1.006,0.918,0.443,0.374,0.372,0.507,0.877,0.895,0.615,0.375,0.371,0.383,0.853,1.003,1.005,1.012,1.03,1.03,1.028,1.027,1.025,1.024,1.023,1.023,1.02,1.019,1.017,1.017,1.015,1.015,1.014,1.012,1.01,1.009,1.008,1.006,1.004]],"rain":[[1.0,1.011,1.024,1.037,1.049,1.06,1.071,1.079,1.084,1.085,1.079,1.057,1.026,0.993,0.954,0.918,0.878,0.835,0.792,0.757,0.732,0.724,0.719,0.726,0.734,0.747,0.765,0.785,0.812,0.834],[1.0,1.003,1.004,0.999,0.996,0.989,0.982,0.97,0.959,0.942,0.92,0.895,0.869,0.84,0.804,0.765,0.717,0.664,0.605,0.543,0.473,0.417,0.365,0.319,0.31,0.352,0.395,0.44,0.487,0.532]],"storm":[[1.0,1.011,1.023,1.031,1.039,1.045,1.041,1.027,1.009,0.978,0.943,0.906,0.864,0.809,0.747,0.687,0.63,0.579,0.526,0.482,0.44,0.408,0.38,0.362,0.344,0.332,0.322,0.315,0.312,0.313],[1.0,1.093,1.099,1.097,1.097,1.095,1.093,1.091,1.09,1.086,1.082,1.077,1.072,1.064,1.038,0.96,0.869,0.765,0.661,0.556,0.451,0.372,0.297,0.233,0.182,0.172,0.167,0.161,0.157,0.153]]};
  // always forward (clouds never run backwards), a little slower than filmed; each film dissolves into the next over several seconds
  const WXSEQ=[{n:'calm'},{n:'wind'},{n:'shade'},{n:'rain'},{n:'calm'},{n:'shade'},{n:'storm'},{n:'wind'}],WXRATE=.8;
  const WXWIND={calm:[1,1],wind:[1,1.9],shade:[1,1.25],rain:[1,1.5],storm:[1,2.1]};
  // iPhone / iPad (and the app's web view) do not reliably give video frames to WebGL: there the films come as frame atlases instead
  const WX_FRAMES=params.has('frames')||/iP(hone|ad|od)/.test(navigator.userAgent)||(navigator.platform==='MacIntel'&&navigator.maxTouchPoints>1);
  const mkVid=()=>{const v=document.createElement('video');v.muted=true;v.defaultMuted=true;v.playsInline=true;v.setAttribute('playsinline','');v.setAttribute('webkit-playsinline','');v.setAttribute('muted','');v.preload='auto';v.loop=false;return v};
  const vids=WX_FRAMES?[]:[mkVid(),mkVid()],vtex=vids.map(v=>{const t=new THREE.VideoTexture(v);t.colorSpace=THREE.NoColorSpace;t.magFilter=THREE.LinearFilter;/* the sharp 1920 clips get mipmaps so the far view does not shimmer */t.minFilter=HQ?THREE.LinearMipmapLinearFilter:THREE.LinearFilter;t.generateMipmaps=HQ;t.anisotropy=HQ?Math.min(8,maxAniso):1;return t});
  // the film must come out of the screen as it was filmed: invert the whole AgX tone map (its colour matrices too, not just its curve)
  const agxInv=(()=>{const N=1024,con=x=>{const x2=x*x,x4=x2*x2;return 15.5*x4*x2-40.14*x4*x+31.96*x4-6.868*x2*x+.4298*x2+.1191*x-.00232};
    const y0=con(0),y1=con(1),hl=new Uint16Array(N*4);
    for(let i=0;i<N;i++){const y=y0+(y1-y0)*i/(N-1);let lo=0,hi=1;for(let k=0;k<50;k++){const m=(lo+hi)/2;if(con(m)<y)lo=m;else hi=m}const v=THREE.DataUtils.toHalfFloat((lo+hi)/2);hl[i*4]=v;hl[i*4+3]=THREE.DataUtils.toHalfFloat(1)}
    // half floats: every phone can filter those (32-bit float textures cannot be filtered on iPhone, which left the window black)
    const t=new THREE.DataTexture(hl,N,1,THREE.RGBAFormat,THREE.HalfFloatType);t.minFilter=t.magFilter=THREE.LinearFilter;t.needsUpdate=true;
    const M=a=>new THREE.Matrix3().set(...a),R2S=M([1.6605,-.5876,-.0728,-.1246,1.1329,-.0083,-.0182,-.1006,1.1187]),S2R=M([.6274,.3293,.0433,.0691,.9195,.0113,.0164,.088,.8956]);
    const IN=M([.856627153315983,.0951212405381588,.0482516061458583,.137318972929847,.761241990602591,.101439036467562,.11189821299995,.0767994186031903,.811302368396859]);
    const OUT=M([1.1271005818144368,-.11060664309660323,-.016493938717834573,-.1413297634984383,1.157823702216272,-.016493938717834257,-.14132976349843826,-.11060664309660294,1.2519364065950405]);
    const A=R2S.clone().invert(),B=OUT.clone().invert(),C=S2R.clone().invert().multiply(IN.clone().invert()).multiplyScalar(1/renderer.toneMappingExposure);
    return {t,A,B,C,y0,y1}})();
  // the fine detail of the scene (brickwork, windows, roof seams, chimneys) from a full-resolution still of the same view, laid over the
  // moving film where nothing moves (clouds and swaying trees are left to the film); it is what makes the view sharp on a phone too
  const DETT=new THREE.TextureLoader().load('tex/wx/detail.webp');DETT.colorSpace=THREE.NoColorSpace;DETT.generateMipmaps=true;DETT.minFilter=THREE.LinearMipmapLinearFilter;DETT.anisotropy=4;
  const BLANK=new THREE.DataTexture(new Uint8Array([201,205,204,255]),1,1);BLANK.needsUpdate=true;
  // each of the two slots shows one film; a slot draws a blend of two pictures (two neighbouring frames, or the video twice)
  const RQ=()=>new THREE.Vector4(0,0,1,1);
  const skyU={tLut:{value:agxInv.t},uA:{value:agxInv.A},uB:{value:agxInv.B},uCm:{value:agxInv.C},uY:{value:new THREE.Vector2(agxInv.y0,agxInv.y1)},uDec:{value:1},tDet:{value:DETT},uDet:{value:1},uYaw0:{value:0},tA0:{value:BLANK},tA1:{value:BLANK},tB0:{value:BLANK},tB1:{value:BLANK},
    rA0:{value:RQ()},rA1:{value:RQ()},rB0:{value:RQ()},rB1:{value:RQ()},fA:{value:0},fB:{value:0},flA:{value:0},flB:{value:0},
    uMix:{value:0},uC:{value:WC.clone()},uExp:{value:1.0},uHalf:{value:Math.tan(36*Math.PI/180)},uPitch:{value:Math.tan(-6*Math.PI/180)}};
  const skyMat=new THREE.ShaderMaterial({uniforms:skyU,side:THREE.BackSide,depthWrite:false,fog:false,
    vertexShader:`varying vec3 vW;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader:`uniform sampler2D tA0,tA1,tB0,tB1,tLut,tDet;uniform float uDet,uYaw0;uniform vec4 rA0,rA1,rB0,rB1;uniform float fA,fB,flA,flB,uMix,uExp,uHalf,uPitch,uDec;uniform vec3 uC;uniform mat3 uA,uB,uCm;uniform vec2 uY;varying vec3 vW;
      vec3 cell(sampler2D t,vec4 r,float fl,vec2 uv){return texture2D(t,vec2(r.x+uv.x*r.z,r.y+mix(uv.y,1.-uv.y,fl)*r.w)).rgb;}

      vec3 dec(vec3 c){return mix(pow(c*.9478672986+.0521327014,vec3(2.4)),c*.0773993808,vec3(lessThanEqual(c,vec3(.04045))));}
      void main(){vec3 d=normalize(vW-uC);float yaw=atan(d.z,d.x)-uYaw0,el=d.y/max(.05,length(d.xz));
        // a wide lens round the window's axis; toward the far sides the film is eased in so it never runs out
        float kx=yaw/atan(uHalf),ky=(el-uPitch)/uHalf;
        float u=.5+.497*tanh(.97*kx),v=.5+.497*tanh(.97*ky);
        vec2 uv=vec2(clamp(u,.002,.998),clamp(v,.003,.997));
        vec3 a=mix(cell(tA0,rA0,flA,uv),cell(tA1,rA1,flA,uv),fA),b=mix(cell(tB0,rB0,flB,uv),cell(tB1,rB1,flB,uv),fB),c=mix(a,b,uMix);
        c=dec(c);c*=exp2((texture2D(tDet,uv).r-.5)*2.*uDet);
        if(ky>1.){c=mix(c,c*vec3(.96,.98,1.04),smoothstep(1.,1.6,ky));}
        c*=uExp;
        // keep the brightest clouds just inside what the tone map can show
        {float k=.9;c=mix(c,k+(.975-k)*(1.-exp(-(c-k)/(.975-k))),step(k,c));}
        // undo the tone map exactly: output transform, curve, log encoding, inset — so the film reaches the screen as filmed
        vec3 w=pow(max(uA*c,0.),vec3(1./2.2));vec3 y=clamp(uB*w,uY.x,uY.y);vec3 q=(y-uY.x)/(uY.y-uY.x);
        vec3 s=vec3(texture2D(tLut,vec2(q.r*(1023./1024.)+.5/1024.,.5)).r,texture2D(tLut,vec2(q.g*(1023./1024.)+.5/1024.,.5)).r,texture2D(tLut,vec2(q.b*(1023./1024.)+.5/1024.,.5)).r);
        c=max(uCm*exp2(s*16.499999-12.47393),0.);
        gl_FragColor=vec4(c,1.);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`});
  const sky=new THREE.Mesh(new THREE.SphereGeometry(38,48,24),skyMat);sky.position.copy(WC);sky.renderOrder=-1;sky.frustumCulled=false;sky.userData.ptSky=true;scene.add(sky);
  // seen from deep in the room, at a slant, the window shows the heart of the view rather than the film's thin edge:
  // the farther you stand, the more the view turns to face you; at the window itself it is exactly where it is
  sky.onBeforeRender=(r,sc,cam)=>{const dx=WC.x-cam.position.x,dz=WC.z-cam.position.z,dist=Math.hypot(dx,dz),k=Math.min(1,Math.max(0,(dist-.6)/1.6))*.82;
    skyU.uYaw0.value=dx>.05?k*Math.max(-1.2,Math.min(1.2,Math.atan2(dz,dx))):0};
  window.__skyU=skyU;
  const groundMat={color:new THREE.Color()};
  const treeMats=[];
  /* the weather player: one clock; slot A plays, five seconds before it ends slot B starts and the picture slowly dissolves across */
  const FADE=6.5;
  const WX={lock:null,pend:false,i:0,slot:0,t:[0,0],fade:-1,ready:[false,false],k:1,kSun:1,wind:1,clip:[null,null],tok:[0,0],stall:0};
  const clipDur=c=>15;
  // --- video source (desktop, Android)
  const WXBLOB={};const wxBlob=n=>WXBLOB[n]||(WXBLOB[n]=fetch((HQ?'tex/wx4k/':'tex/wx/')+n+'.mp4').then(async r=>{if(!r.ok)throw new Error(r.status);
    const tot=+r.headers.get('content-length')||0;if(!r.body||!r.body.getReader||!tot)return r.blob();
    const rd=r.body.getReader(),parts=[];let got=0;for(;;){const {done,value}=await rd.read();if(done)break;parts.push(value);got+=value.length;window.__loadVid&&window.__loadVid(n,got,tot)}
    return new Blob(parts,{type:'video/mp4'})}).then(b=>URL.createObjectURL(b)));
  // --- frame source (iPhone): 3x3 atlases of 512 px frames at 6 per second, decoded off the main thread, a few kept at a time
  const ATL=new Map();
  function atlas(n,a){const key=n+a;let e=ATL.get(key);if(e){e.used=performance.now();return e.tex}
    e={tex:null,used:performance.now()};ATL.set(key,e);
    fetch('tex/wxg/'+n+'_'+(a+1)+'.webp').then(r=>r.blob()).then(b=>createImageBitmap(b)).then(bm=>{const t=new THREE.Texture(bm);t.flipY=false;t.colorSpace=THREE.NoColorSpace;t.minFilter=THREE.LinearFilter;t.generateMipmaps=false;t.needsUpdate=true;e.tex=t})
      .catch(err=>{console.warn('weather frames',key,err);ATL.delete(key)});
    if(ATL.size>8){let old=null;for(const [k,v] of ATL)if(v.tex&&(!old||v.used<old[1].used))old=[k,v];if(old){old[1].tex.dispose();if(old[1].tex.image&&old[1].tex.image.close)old[1].tex.image.close();ATL.delete(old[0])}}
    return null}
  function cue(slot,idx){const c=WX.lock?{n:WX.lock}:WXSEQ[idx%WXSEQ.length];WX.clip[slot]=c;WX.ready[slot]=WX_FRAMES;WX.t[slot]=0;
    if(WX_FRAMES){atlas(c.n,0);atlas(c.n,1);return}
    const v=vids[slot],tok=(WX.tok[slot]=(WX.tok[slot]||0)+1);
    wxBlob(c.n).then(url=>{if(WX.tok[slot]!==tok)return;v.onloadeddata=()=>{if(WX.tok[slot]!==tok)return;WX.ready[slot]=true;v.playbackRate=WXRATE;if(slot===WX.slot&&!STILL)v.play().catch(()=>{})};v.defaultPlaybackRate=WXRATE;v.src=url;v.load()}).catch(e=>console.warn('weather film',c.n,e))}
  cue(0,0);cue(1,1);
  // put slot s's picture for its clip time into the sky uniforms
  const WFR=[[skyU.tA0,skyU.tA1,skyU.rA0,skyU.rA1,skyU.fA,skyU.flA],[skyU.tB0,skyU.tB1,skyU.rB0,skyU.rB1,skyU.fB,skyU.flB]];
  function showSlot(s){const U=WFR[s],c=WX.clip[s];if(!c)return;
    if(!WX_FRAMES){if(WX.ready[s]){U[0].value=U[1].value=vtex[s];U[2].value.set(0,0,1,1);U[3].value.set(0,0,1,1);U[4].value=0;U[5].value=0}return}
    const t=WX.t[s],tt=Math.min(14.99,t),ft=Math.max(0,Math.min(59,tt*4)),f0=Math.floor(ft),f1=Math.min(59,f0+1);
    const a0=atlas(c.n,(f0/16)|0),a1=atlas(c.n,(f1/16)|0);atlas(c.n,Math.min(3,((f1/16)|0)+1));
    const rect=(f,r)=>{const cc=f%16;r.set((cc%4)/4,((cc/4)|0)/4,1/4,1/4)};
    if(a0){U[0].value=a0;rect(f0,U[2].value);U[1].value=a1||a0;rect(a1?f1:f0,U[3].value);U[4].value=a1?ft-f0:0;U[5].value=1}}
  // how bright the sky and the sunlit roofs are in a clip at time t, relative to its first frame
  function wxLight(c,t){if(!c)return [1,1];const C=WXC[c.n];let tt=Math.min(15,t);const f=Math.min(C[0].length-1.001,Math.max(0,tt*2)),i=Math.floor(f),r=f-i;
    return [C[0][i]*(1-r)+C[0][i+1]*r,C[1][i]*(1-r)+C[1][i+1]*r]}
  function wxStep(dt,still){
    const s=WX.slot,o=1-s,c=WX.clip[s];
    // the clock: the film's own clock while a video plays, otherwise real time
    const adv=(k)=>{if(!WX.clip[k])return;if(!WX_FRAMES&&!still){const v=vids[k];if(WX.ready[k]&&!v.paused&&v.currentTime>0){WX.t[k]=v.currentTime;return}}WX.t[k]+=still?0:dt*WXRATE};
    if(WX.ready[s]||WX_FRAMES)adv(s);
    if(WX.fade>=0)adv(o);
    if(!WX_FRAMES&&!still){const v=vids[s];if(WX.ready[s]&&v.paused&&!v.ended&&WX.t[s]<clipDur(c)-.2){v.playbackRate=WXRATE;v.play().catch(()=>{})}}
    // a weather chosen from the menu: once the next film is ready, dissolve to it straight away
    if(WX.pend&&WX.fade<0&&!still){const oc=WX.clip[o],rd=WX_FRAMES?!!(oc&&ATL.get(oc.n+'0')&&ATL.get(oc.n+'0').tex):WX.ready[o];if(rd){WX.pend=false;WX.t[s]=Math.max(WX.t[s],clipDur(c)-FADE)}}
    if(WX.fade<0&&WX.t[s]>=clipDur(c)-FADE){WX.fade=0;WX.t[o]=0;if(!WX_FRAMES&&!still&&WX.ready[o]){const nv=vids[o];nv.currentTime=0;nv.playbackRate=WXRATE;nv.play().catch(()=>{})}if(!WX_FRAMES)wxBlob(WXSEQ[(WX.i+2)%WXSEQ.length].n).catch(()=>{})}
    if(WX.fade>=0){WX.fade=still?WX.fade:(!WX_FRAMES&&WX.ready[o]&&!vids[o].paused?Math.max(WX.fade,WX.t[o]):WX.fade+dt);const m=Math.min(1,WX.fade/FADE),e=m*m*(3-2*m);skyU.uMix.value=s===0?e:1-e;
      if(m>=1){WX.fade=-1;WX.slot=o;WX.i++;if(!WX_FRAMES)vids[s].pause();cue(s,WX.i+1);if(WX.pendAfter){WX.pendAfter=false;WX.pend=true}}}
    showSlot(0);showSlot(1);
    // the room follows the light outside
    const L1=wxLight(WX.clip[0],WX.t[0]),L2=wxLight(WX.clip[1],WX.t[1]);
    const DK={calm:1,wind:1,shade:1,rain:.6,storm:.5},dk=c=>c?DK[c.n]:1;skyU.uDet.value=(dk(WX.clip[0])*(1-skyU.uMix.value)+dk(WX.clip[1])*skyU.uMix.value)*(WX_FRAMES?1:.6);
    const m=skyU.uMix.value,ks=L1[0]*(1-m)+L2[0]*m,kr=L1[1]*(1-m)+L2[1]*m;WX.k=.35+.65*Math.min(1.1,ks);WX.kSun=Math.min(1.1,kr);
    const pr=(k)=>Math.min(1,WX.t[k]/15);
    const wf=k=>{const W=WXWIND[WX.clip[k]?WX.clip[k].n:'calm'];return W[0]+(W[1]-W[0])*pr(k)};WX.wind=wf(0)*(1-m)+wf(1)*m}
  /* sound: each weather's own sound as a seamless loop, all running at once, their levels following what is in the window — never a gap.
     It comes in through the window: muffled and quieter when it is shut. Browsers only allow sound after a first touch or key. */
  const AU={ctx:null,src:{},g:{},buf:{},lp:null,pan:null,master:null,inner:null,on:true,loading:false,unlock:null,bed:{}};
  function silentWav(){const sr=8000,n=800,b=new ArrayBuffer(44+n*2),d=new DataView(b);const w=(o,s)=>{for(let i=0;i<s.length;i++)d.setUint8(o+i,s.charCodeAt(i))};
    w(0,'RIFF');d.setUint32(4,36+n*2,true);w(8,'WAVE');w(12,'fmt ');d.setUint32(16,16,true);d.setUint16(20,1,true);d.setUint16(22,1,true);d.setUint32(24,sr,true);d.setUint32(28,sr*2,true);d.setUint16(32,2,true);d.setUint16(34,16,true);w(36,'data');d.setUint32(40,n*2,true);
    return URL.createObjectURL(new Blob([b],{type:'audio/wav'}))}
  const loadBuf=(ctx,url)=>fetch(url).then(r=>{if(!r.ok)throw new Error(r.status);return r.arrayBuffer()}).then(ab=>new Promise((res,rej)=>{const p=ctx.decodeAudioData(ab,res,rej);if(p&&p.then)p.then(res,rej)}));
  // a seamless loop of a decoded buffer (AAC adds a few ms of padding at the start: loop exactly the real length)
  function startLoop(ctx,buf,len,dest){const s=ctx.createBufferSource(),g=ctx.createGain();s.buffer=buf;s.loop=true;g.gain.value=0;s.connect(g);g.connect(dest);s.start(0,Math.random()*buf.duration*.9);return {s,g}}
  function audioStart(){if(STILL)return;
    try{if(navigator.audioSession)navigator.audioSession.type='playback'}catch(e){}
    // iPhone with the side switch on silent: a playing <audio> element puts the page into "playback" mode, so it is heard anyway
    if(!AU.unlock){const a=AU.unlock=new Audio(silentWav());a.loop=true;a.setAttribute('playsinline','');a.volume=.01}
    AU.unlock.play().catch(()=>{});
    if(!AU.ctx){const C=window.AudioContext||window.webkitAudioContext;if(!C)return;const ctx=AU.ctx=new C();
      // outside -> (window: muffled when shut) -> left/right toward the window -> volume
      AU.lp=ctx.createBiquadFilter();AU.lp.type='lowpass';AU.lp.frequency.value=900;AU.lp.Q.value=.5;
      AU.pan=ctx.createStereoPanner?ctx.createStereoPanner():ctx.createGain();AU.master=ctx.createGain();AU.master.gain.value=0;
      AU.lp.connect(AU.pan);AU.pan.connect(AU.master);AU.master.connect(ctx.destination);
      AU.inner=ctx.createGain();AU.inner.gain.value=.9;AU.inner.connect(AU.pan)}
    if(AU.ctx.state!=='running')AU.ctx.resume().catch(()=>{});
    if(!AU.loading){AU.loading=true;const ctx=AU.ctx;
      // every sound in one file, cut apart here: the weathers' own sounds, the street, the birds, the window, a gust
      const SEG={"total":15210490,"seg":{"calm":[0,573300],"wind":[573300,573300],"shade":[1146600,573300],"rain":[1719900,573300],"storm":[2293200,573300],"city":[2866500,6438600],"birds":[9305100,5234670],"open":[14539770,378880],"gust":[14918650,291840]}};
      loadBuf(ctx,'tex/wx/audio.mp4').then(all=>{const sr=all.sampleRate,k=sr/44100,nc=all.numberOfChannels;
        // the phone/computer may decode at 48 kHz: compare lengths at that rate. AAC may (Safari) or may not (Chrome) keep its 2112-sample lead-in
        const extra=all.length-Math.round(SEG.total*k),pad=extra>1500*k?Math.min(Math.round(2112*k),extra):0;
        // each piece is cut a little inside its edges (no sliver of its neighbour can ever get in), and a loop is closed by
        // cross-fading its own tail into its head over a couple of seconds — so it runs round for ever with no seam to hear
        const piece=(n,trim)=>{const [a,len]=SEG.seg[n],s0=pad+Math.round(a*k)+trim,L=Math.min(Math.round(len*k)-2*trim,all.length-s0);return {s0,L}};
        const cut=n=>{const {s0,L}=piece(n,Math.round(.012*sr)),b=ctx.createBuffer(nc,L,sr),F=Math.round(.006*sr);
          for(let c=0;c<nc;c++){const src=all.getChannelData(c),d=b.getChannelData(c);d.set(src.subarray(s0,s0+L));for(let i=0;i<F;i++){const w=i/F;d[i]*=w;d[L-1-i]*=w}}return b};
        const loop=n=>{const {s0,L}=piece(n,Math.round(.15*sr)),X=Math.round(Math.min(2.5*sr,L*.2)),N=L-X,b=ctx.createBuffer(nc,N,sr);
          for(let c=0;c<nc;c++){const src=all.getChannelData(c).subarray(s0,s0+L),d=b.getChannelData(c);d.set(src.subarray(0,N));
            for(let i=0;i<X;i++){const t=(i+.5)/X*Math.PI/2;d[i]=src[i]*Math.sin(t)+src[N+i]*Math.cos(t)}}return b};
        for(const n of Object.keys(WXWIND)){const L=startLoop(ctx,loop(n),99,AU.lp);AU.src[n]=L.s;AU.g[n]=L.g}
        AU.bed.city=startLoop(ctx,loop('city'),99,AU.lp);AU.bed.birds=startLoop(ctx,loop('birds'),99,AU.lp);AU.buf.open=cut('open');AU.buf.gust=cut('gust');AU.pad=pad}).catch(e=>console.warn('sound',e));
      // the room's own sounds, decoded ahead so each plays the instant its event happens: glass, the stone, paper in the draught
      for(const n of ['glass','stone','knock'])loadBuf(ctx,'tex/sfx/'+n+'.wav').then(b=>{AU.buf[n]=b}).catch(e=>console.warn('sound',n,e));
      loadBuf(ctx,'tex/sfx/paper.m4a').then(b=>{const sr=b.sampleRate,nc=b.numberOfChannels,X=Math.round(3*sr),N=b.length-X,o=ctx.createBuffer(nc,N,sr);
        for(let c=0;c<nc;c++){const src=b.getChannelData(c),d=o.getChannelData(c);d.set(src.subarray(0,N));for(let i=0;i<X;i++){const t=(i+.5)/X*Math.PI/2;d[i]=src[i]*Math.sin(t)+src[N+i]*Math.cos(t)}}
        AU.bed.paper=startLoop(ctx,o,99,AU.inner)}).catch(e=>console.warn('sound paper',e))}}
  // one-off sounds: the window's catch and hinges (in the room), a gust (from outside)
  function sfx(name,vol,outside,delay,rate){if(!AU.ctx||!AU.on||!AU.buf[name])return;const s=AU.ctx.createBufferSource(),g=AU.ctx.createGain();s.buffer=AU.buf[name];if(rate)s.playbackRate.value=rate;g.gain.value=vol;s.connect(g);g.connect(outside?AU.lp:AU.inner);s.start(AU.ctx.currentTime+(delay||0))}
  // the story's sounds land on the frame of their event; in a film's slow motion they are slowed with it
  const sRate=()=>typeof DIR!=='undefined'&&DIR.on&&DIR.ws<.9?Math.max(.62,.5+DIR.ws):1;
  window.addEventListener('glass',()=>sfx('glass',1,false,0,sRate()));
  window.addEventListener('stoneland',e=>{if(!AU.landed){AU.landed=true;sfx('stone',Math.min(1,.35+(e.detail?e.detail.v:5)/9),false,0,sRate())}});
  window.addEventListener('knock',e=>{if(e.detail&&e.detail.roll)sfx('knock',.32,false,0,.72*sRate());else sfx('knock',.85,false,0,sRate())});
  for(const ev of ['pointerdown','touchend','keydown','click'])window.addEventListener(ev,()=>{if(AU.on)audioStart()},{capture:true,passive:true});
  document.addEventListener('visibilitychange',()=>{if(!AU.ctx)return;if(document.hidden)AU.ctx.suspend().catch(()=>{});else if(AU.on)AU.ctx.resume().catch(()=>{})});
  // the street is alive: its level drifts, now and then something passes close; birds come and go, and fall silent in rain
  const AUW=new THREE.Vector3(),AUF=new THREE.Vector3();
  function audioStep(){if(!AU.ctx)return;const m=skyU.uMix.value,t=AU.ctx.currentTime,o=Math.min(1,S.openFrac||0),T=S.time;
    const w={};for(const n of Object.keys(WXWIND))w[n]=0;
    if(WX.clip[0])w[WX.clip[0].n]+=Math.cos(m*Math.PI/2);if(WX.clip[1])w[WX.clip[1].n]+=Math.sin(m*Math.PI/2);
    for(const n in AU.g)AU.g[n].gain.setTargetAtTime(Math.min(1,w[n])*.75,t,.4);
    const wet=Math.min(1,w.rain+w.storm),windy=Math.min(1,w.wind+w.storm*.5);
    if(AU.bed.city){const pass=Math.max(0,Math.sin(T*.21+1.3)*Math.sin(T*.057+.4))*.5,drift=.75+.25*Math.sin(T*.031)*Math.sin(T*.017+2);
      AU.bed.city.g.gain.setTargetAtTime(.16*drift*(1+pass)*(1-.35*wet),t,.8)}
    if(AU.bed.paper){const flying=(S.flyN||0)*.18;AU.bed.paper.g.gain.setTargetAtTime(Math.min(.95,Math.max(0,S.interior-.22)*.8+S.gust*.25+flying),t,.18)}
    if(AU.bed.birds){const come=.55+.45*Math.sin(T*.043+.7)*Math.sin(T*.11);AU.bed.birds.g.gain.setTargetAtTime(.22*come*(1-.9*wet)*(1-.4*windy),t,1.2)}
    // where is the window from where you stand and look?
    camera.getWorldDirection(AUF);AUW.set(WC.x-camera.position.x,0,WC.z-camera.position.z);const dist=AUW.length();AUW.normalize();
    const right=AUF.z*AUW.x-AUF.x*AUW.z;if(AU.pan.pan)AU.pan.pan.setTargetAtTime(Math.max(-.85,Math.min(.85,-right*.9)),t,.1);
    AU.lp.frequency.setTargetAtTime(650+17000*Math.pow(o,1.6),t,.15);
    AU.master.gain.setTargetAtTime(AU.on?(AU.vol??.8)/.8*(.4+.6*o)*(S.taped?.55:1)/(1+.18*dist):0,t,.15)}
  window.__au=()=>AU.ctx?[AU.ctx.state,Object.keys(AU.bed),Object.keys(AU.buf),Object.fromEntries(Object.entries(AU.g).map(([k,g])=>[k,+g.gain.value.toFixed(2)])),+AU.master.gain.value.toFixed(2),AU.ctx.sampleRate,AU.pad,AU.bed.city&&+AU.bed.city.s.buffer.duration.toFixed(2)]:'none';
  window.__wx=()=>({frames:WX_FRAMES,slot:WX.slot,mix:+skyU.uMix.value.toFixed(3),t:WX.t.map(v=>+v.toFixed(2)),clip:WX.clip.map(c=>c&&c.n),ready:WX.ready,fade:+WX.fade.toFixed(2),i:WX.i,vids:vids.map(v=>[+v.currentTime.toFixed(2),v.paused]),atl:[...ATL.keys()].join(',')});
  // stills: put the film at a given clip and time and wait for that picture
  async function wxSeek(idx,t){WX.i=idx;WX.slot=0;cue(0,idx);cue(1,idx+1);skyU.uMix.value=0;WX.fade=-1;WX.t[0]=t;
    if(WX_FRAMES){for(let k=0;k<200;k++){showSlot(0);if(skyU.tA0.value!==BLANK&&skyU.tA1.value.image)break;await new Promise(r=>setTimeout(r,50))}wxStep(0,true);return}
    for(let k=0;k<400&&!WX.ready[0];k++)await new Promise(r=>setTimeout(r,50));
    await new Promise(r=>{const v=vids[0];v.onseeked=()=>r();v.currentTime=Math.max(.01,t);setTimeout(r,20000)});
    WX.t[0]=t;await new Promise(r=>setTimeout(r,150));vtex[0].needsUpdate=true;wxStep(0,true)}
  /* door */
  const dW=.8,dH=2.15,dT=.045;
  const dm=new THREE.MeshStandardMaterial({map:tex(tDoor.map,{srgb:true}),bumpMap:tex(tDoor.bump),bumpScale:2.5,roughness:.85});
  if(hasT('door')){const n=TX.doorN.clone();n.needsUpdate=true;n.repeat.set(1,dH/dW);n.offset.y=-((dH/dW)%1);const r=TX.doorR.clone();r.needsUpdate=true;r.repeat.set(1,dH/dW);r.offset.y=n.offset.y;dm.normalMap=n;dm.roughnessMap=r;dm.roughness=1;dm.bumpMap=null;dm.needsUpdate=true}
  const dEdgeT=hasT('door')?pbr('door',[.06,2]):null;
  const dEdge=new THREE.MeshStandardMaterial({color:'#7d6a56',roughness:.9});
  const dE=dEdgeT||dEdge,door=new THREE.Mesh(new THREE.BoxGeometry(dW,dH,dT),[dE,dE,dE,dE,dm,dE]);
  door.position.set(0,dH/2,0);door.rotation.y=Math.PI/2;door.castShadow=true;door.receiveShadow=true;
  const lean=.19,dg=new THREE.Group();dg.add(door);dg.rotation.z=lean;
  dg.position.set(X0+.02+dT*.5+dH*Math.sin(lean),0,-1.72);scene.add(dg);
  { const dx=dg.position.x;OCC.uCapA.value[0].set(dx,-2.1,dx,-1.34);OCC.uCapB.value[0].set(.045,.55,0,0);
    OCC.uOcc.value[6].set(dx-.2,-1.95,.26,.22);OCC.uOcc.value[7].set(dx-.2,-1.5,.26,.22)}

  /* paper wall */
  const PW=.297,PH=.42;
  const PGX=14,PGY=20;
  function paperGeo(curl,bow,twist,seed,flat){
    const g=new THREE.PlaneGeometry(PW,PH,PGX,PGY);g.translate(0,-PH/2,0);
    const p=g.attributes.position,r=mulberry32(seed),dog=r()<.55?(r()<.5?-1:1):0,dogA=(.012+r()*.03)*(flat?.35:1),ph=r()*6.28;
    // an old fold: a straight crease somewhere across (or along) the sheet, the two halves meeting at a shallow angle
    const fold=r()<.6,fa=(r()-.5)*.5+(r()<.25?Math.PI/2:0),fc=Math.cos(fa),fs=Math.sin(fa),fo=(r()-.5)*.16,fA=(r()<.5?1:-1)*(.003+r()*.006)*(flat?.6:1);
    const cn=makeNoise(seed*7+3);
    for(let i=0;i<p.count;i++){const x=p.getX(i),y=p.getY(i),t=-y/PH,xn=x/(PW/2);
      const c=dog?Math.max(0,(xn*dog*.5+t*.5)-.8)/.2:0;
      let z=curl*t*t*1.3+bow*(1-xn*xn)*t+twist*xn*t*t+dogA*c*c+.0035*Math.sin(xn*3.1+ph)*t*t*t*(flat?.3:1);
      if(fold){const d=x*fs+(y+PH/2)*fc-fo;z+=fA*Math.max(0,1-Math.abs(d)/.09)}
      z+=(cn.n(x*22+ph,y*22)-.5)*.0026+(cn.n(x*60,y*60+ph)-.5)*.0008;   // cockle
      p.setZ(i,flat?z:Math.max(z,0))}
    g.computeVertexNormals();return g}
  const vr=mulberry32(101),variants=[],flatVariants=[];
  for(let i=0;i<16;i++){const c=vr()*.032,b=(vr()-.4)*.018,tw=(vr()-.5)*.034;variants.push(paperGeo(c,b,tw,200+i,false));flatVariants.push(paperGeo(c*.12,b*.15,tw*.15,200+i,true))}
  const r=mulberry32(42),sheets=[];
  const cols=12,xa=X0+.2,xb=X1-.14,step=(xb-xa)/(cols-1);
  for(let row=0;row<10;row++){const yTop=RH-.28-row*.272;
    for(let c=0;c<cols;c++){const n=1+(r()<.75?1:0);
      for(let k=0;k<n;k++){
        let x=xa+c*step+(r()-.5)*.12+(row%2)*step*.4+(k?(r()-.5)*.2:0);x=Math.min(xb,Math.max(xa,x));
        const y=yTop+(r()-.5)*.07-(k?.06+r()*.08:0);
        let lean=.01+r()*.035;
        if(r()<.08)lean=.1+r()*.2;
        if(row>=8)lean+=.04+r()*.22;
        const z=Z0+.004+(9-row)*.009+k*.005+r()*.003;   // shingled: each row hangs over the top of the row below
        const d=Math.hypot(x-WC.x,y-WC.y,z-WC.z);
        sheets.push({row,anim:r()<.45?0:(r()<.55?1:2),v:Math.floor(r()*16),x,y,z,rz:(r()-.5)*.3,lean,prox:Math.exp(-d/1.4),f:.8+r()*1.1,ph:r()*6.28,hidden:false,
          tapes:r()<.78?[[0,0,(r()-.5)*.35]]:[[-.12,0,.55+r()*.3],[.12,0,-.55-r()*.3]]});
        {const s=sheets[sheets.length-1];if(row<9&&r()<.3){const sd=r()<.5?-1:1;s.tapes.push([sd*(.09+r()*.05),-PH+.025,sd*(.5+r()*.5)]);s.botTape=true}}
      }}}
  // a sheet whose bottom is overlapped by the sheet below cannot flap: only uncovered sheets and the bottom rows move freely
  // a sheet with another sheet lying over it cannot curl, lean or flap into it: the gap in front sets how much it may move
  const VMAX=variants.map(g=>{let m=0;const p=g.attributes.position;for(let i=0;i<p.count;i++)m=Math.max(m,p.getZ(i));return m});
  function updateCover(){
    for(const s of sheets){if(s.hidden)continue;let gap=1e9;
      for(const o of sheets){if(o===s||o.hidden||o.z<=s.z)continue;if(Math.abs(o.x-s.x)<PW-.01&&o.y>s.y-PH+.01&&o.y-PH<s.y-PH*.4)gap=Math.min(gap,o.z-s.z)}
      s.gap=gap;if(s.botTape){s.curl=.08;s.leanMax=0;s.free=0}else if(gap>1){s.curl=1;s.leanMax=1;s.free=1}
      else{s.curl=THREE.MathUtils.clamp((gap-.0012)/(VMAX[s.v]+1e-4),.04,1);s.leanMax=Math.max(0,gap-.0012-VMAX[s.v]*s.curl)/PH;s.free=THREE.MathUtils.clamp((gap-.0015)/.07,0,.2)}}
    meshes&&meshes.forEach((m,i)=>{const fa=m.geometry.attributes.aFree,ca=m.geometry.attributes.aCurl;byV[i].forEach((s,k)=>{fa.array[k]=s.free*(1-(s.press||0));ca.array[k]=s.curl*(1-.9*(s.press||0))});fa.needsUpdate=true;ca.needsUpdate=true})}
  let meshes=null;
  const byV=variants.map(()=>[]);sheets.forEach(s=>{s.idx=byV[s.v].length;byV[s.v].push(s);s.m=new THREE.Matrix4()});
  updateCover();
  const pcol=new THREE.Color(),pr=mulberry32(77);
  meshes=variants.map((g0,i)=>{const g=g0.clone(),fa=new Float32Array(Math.max(1,byV[i].length)),ca=new Float32Array(Math.max(1,byV[i].length));byV[i].forEach((s,k)=>{fa[k]=s.free;ca[k]=s.curl});g.setAttribute('aFree',new THREE.InstancedBufferAttribute(fa,1));g.setAttribute('aCurl',new THREE.InstancedBufferAttribute(ca,1));const m=new THREE.InstancedMesh(g,pm,Math.max(1,byV[i].length));m.count=byV[i].length;m.castShadow=true;m.receiveShadow=true;m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    for(let k=0;k<m.count;k++){const u=pr();if(u<.62)pcol.setRGB(1,.992,.972);else if(u<.86)pcol.setRGB(.975,.955,.905);else if(u<.95)pcol.setRGB(.93,.925,.9);else pcol.setRGB(.96,.93,.86);
      pcol.multiplyScalar(.97+pr()*.05);m.setColorAt(k,pcol)}
    scene.add(m);return m});
  // the left casement swings toward the papered wall: find where it first meets the paper; past that it presses the sheets flat
  const HX=leafA.hx,cornerSheets=sheets.filter(s=>s.x>.9);
  const leafZAt=(th,x)=>{const sn=Math.sin(th);if(sn<1e-3)return 1e9;const t=(HX-x)/sn;if(t<0||t>lw)return 1e9;return leafA.hingeZ+t*Math.cos(th)};
  leafA.soft=leafA.max;
  for(let th=1.2;th<leafA.max;th+=.003){let hit=false;for(const s of cornerSheets){for(const dx of [-PW/2,0,PW/2]){const zl=leafZAt(th,s.x+dx);if(zl<s.z+.02+.03*s.curl){hit=true;break}}if(hit)break}if(hit){leafA.soft=th;break}}
  function applyPress(){const th=leafA.th;let changed=false;
    for(const s of cornerSheets){let p=0;if(!s.hidden&&th>leafA.soft-.02&&s.y>WIN.y0-.1&&s.y-PH<WIN.y1){let zl=1e9;for(const dx of [-PW/2,0,PW/2])zl=Math.min(zl,leafZAt(th,s.x+dx));
        if(zl<1e8){const top=s.z+.02+.03*s.curl;p=THREE.MathUtils.clamp((top-zl)/Math.max(.01,top-(Z0+.006)),0,1)}}
      if(Math.abs(p-(s.press||0))>.01){s.press=p;changed=true}}
    if(changed){meshes.forEach((m,i)=>{const fa=m.geometry.attributes.aFree,ca=m.geometry.attributes.aCurl;byV[i].forEach((s,k)=>{fa.array[k]=s.free*(1-(s.press||0));ca.array[k]=s.curl*(1-.9*(s.press||0))});fa.needsUpdate=true;ca.needsUpdate=true})}}
  const tapeList=[];sheets.forEach(s=>s.tapes.forEach(t=>tapeList.push({s,t})));
  const tapes=new THREE.InstancedMesh(new THREE.PlaneGeometry(.042,.085),tm,tapeList.length);tapes.receiveShadow=true;tapes.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(tapes);
  const tapeLocal=tapeList.map(({t})=>{const o=new THREE.Object3D();o.position.set(t[0],t[1],t[1]<-.1?.0035:.0022);o.rotation.z=t[2];o.updateMatrix();return o.matrix.clone()});
  const dummy=new THREE.Object3D(),tmp=new THREE.Matrix4(),ZERO=new THREE.Matrix4().makeScale(0,0,0);
  function placeSheets(){
    const w=S.interior,t=S.time;
    for(const s of sheets){
      if(s.hidden){meshes[s.v].setMatrixAt(s.idx,ZERO);continue}
      const amp=(.004+.075*s.prox*(s.lean>.1?1.5:1))*w*s.free;
      let sway;
      if(s.anim===0)sway=amp*(.5+.5*Math.sin(t*s.f*(1+w*.8)+s.ph))*(.75+.25*Math.sin(t*.37+s.x*2))+amp*.25*Math.sin(t*s.f*3.3+s.ph*2)*s.prox;
      else if(s.anim===1)sway=amp*1.2*(.5+.5*Math.sin(t*s.f*.45+s.ph))*(.5+.5*Math.sin(t*.21+s.ph*2));     // slow breathing
      else sway=amp*.45*(.5+.5*Math.sin(t*s.f*6.5+s.ph))*(.4+.6*Math.max(0,Math.sin(t*.8+s.ph)));      // trembling in gusts
      const ps=s.press||0,lean=Math.min(s.lean,s.leanMax)*(1-ps),sw=Math.min(Math.max(0,sway),Math.max(0,s.leanMax-lean))*(1-ps);
      dummy.position.set(s.x,s.y,s.z);dummy.rotation.set(-(lean+sw),0,s.rz+sw*.25*Math.sin(s.ph),'ZYX');
      dummy.updateMatrix();s.m.copy(dummy.matrix);meshes[s.v].setMatrixAt(s.idx,s.m)}
    meshes.forEach(m=>m.instanceMatrix.needsUpdate=true);
    tapeList.forEach((e,i)=>{tmp.multiplyMatrices(e.s.m,tapeLocal[i]);tapes.setMatrixAt(i,tmp)});tapes.instanceMatrix.needsUpdate=true;
  }

  /* loose sheets: on the floor, and the ones the wind tears off the wall */
  const FN=new THREE.Vector3(),FR=new THREE.Vector3(),FT=new THREE.Vector3(),FA=new THREE.Vector3();
  const flyers=[],V=new THREE.Vector3(),W=new THREE.Vector3(),Q=new THREE.Quaternion(),E=new THREE.Euler(),A=new THREE.Vector3();
  const pmTones=[];
  function makeFlyer(v){const m=new THREE.Mesh(variants[v],pm);m.castShadow=true;m.receiveShadow=true;m.visible=false;scene.add(m);
    const f={m,v,vel:new THREE.Vector3(),ang:new THREE.Vector3(),rest:false,active:false,ph:simRand()*6.28,restY:.004,target:new THREE.Quaternion(),wall:false};flyers.push(f);return f}
  function flatTarget(f){V.set(0,-1,0).applyQuaternion(f.m.quaternion);const up=A.set(0,0,1).applyQuaternion(f.m.quaternion).y>=0;
    if(up)E.set(-Math.PI/2-.012,Math.atan2(V.x,V.z),0,'YXZ');else E.set(Math.PI/2+.012,Math.atan2(-V.x,-V.z),0,'YXZ');f.target.setFromEuler(E)}
  const floorInit=[];
  for(let i=0;i<34;i++){const f=makeFlyer(Math.floor(r()*16));const x=X0+.25+r()*(RW-.45),z=Z0+.08+Math.pow(r(),1.6)*.75,ry=r()*6.28;
    f.restY=.003+i*.0012;floorInit.push({f,x,z,ry})}
  // height of a sheet variant's surface at a local point (bilinear on its 11x15 grid)
  const VZ=flatVariants.map(g=>g.attributes.position),NX1=PGX+1;
  const varZ=(v,lx,ly)=>{const p=VZ[v],fx=Math.min(PGX-.001,Math.max(0,(lx/PW+.5)*PGX)),fy=Math.min(PGY-.001,Math.max(0,-ly/PH*PGY)),ix=Math.floor(fx),iy=Math.floor(fy),tx=fx-ix,ty=fy-iy,k=iy*NX1+ix;
    return (p.getZ(k)*(1-tx)+p.getZ(k+1)*tx)*(1-ty)+(p.getZ(k+NX1)*(1-tx)+p.getZ(k+NX1+1)*tx)*ty};
  const rhP=new THREE.Vector3(),rhL=new THREE.Vector3(),rhQ=new THREE.Quaternion();
  // lowest pivot height that keeps every part of the (curled) sheet above the planks and above sheets already lying there
  function restHeight(f,x,z,q){const s=A.set(0,0,1).applyQuaternion(q).y>=0?1:-1;let need=-1;
    for(let i=0;i<5;i++)for(let j=0;j<7;j++){const lx=(i/4-.5)*(PW-.02),ly=-.01-j/6*(PH-.02);
      rhP.set(lx,ly,0).applyQuaternion(q);const px=x+rhP.x,pz=z+rhP.z;
      let under=paperFloor(px,pz)+.0016;
      for(const o of flyers){if(o===f||!o.active||!o.rest||!o.m.visible||(o.restOrder||0)>(f.restOrder||1e9))continue;const op=o.m.position;if(Math.abs(op.x-px)>.55||Math.abs(op.z-pz)>.55)continue;
        rhL.set(px-op.x,0,pz-op.z).applyQuaternion(rhQ.copy(o.m.quaternion).invert());
        if(rhL.x<-PW/2||rhL.x>PW/2||rhL.y>0||rhL.y<-PH)continue;
        const os=A.set(0,0,1).applyQuaternion(o.m.quaternion).y>=0?1:-1;under=Math.max(under,op.y+os*varZ(o.v,rhL.x,rhL.y)+.0019)}
      need=Math.max(need,under-s*varZ(f.v,lx,ly))}
    return need}
  // a sheet lying still settles into whatever is under it: planks, other sheets. Paper bridges a little (it is stiff over a couple of centimetres) but never floats or cuts through
  let restCounter=100,drapeBudget=0;
  const zOf=(g,lx,ly)=>{const p=g.attributes.position,fx=Math.min(PGX-.001,Math.max(0,(lx/PW+.5)*PGX)),fy=Math.min(PGY-.001,Math.max(0,-ly/PH*PGY)),ix=Math.floor(fx),iy=Math.floor(fy),tx=fx-ix,ty=fy-iy,k=iy*NX1+ix;
    return (p.getZ(k)*(1-tx)+p.getZ(k+1)*tx)*(1-ty)+(p.getZ(k+NX1)*(1-tx)+p.getZ(k+NX1+1)*tx)*ty};
  const drU=new Float32Array((PGX+1)*(PGY+1)),drH=new Float32Array(drU.length),drT=new Float32Array(drU.length),drQ=new THREE.Quaternion();
  function drape(f){if(!f.dg||f.dgv!==f.v){if(f.dg)f.dg.dispose();f.dg=flatVariants[f.v].clone();f.dgv=f.v}
    const g=f.dg,P=g.attributes.position,B=flatVariants[f.v].attributes.position,q=f.m.quaternion,pos=f.m.position,s=A.set(0,0,1).applyQuaternion(q).y>=0?1:-1;
    const below=flyers.filter(o=>o!==f&&o.active&&o.rest&&o.m.visible&&(o.restOrder||0)<(f.restOrder||0)&&Math.abs(o.m.position.x-pos.x)<.6&&Math.abs(o.m.position.z-pos.z)<.6);
    // support under each point of the sheet: the highest plank or sheet anywhere within its own cell (the sheet is flat between its points, the floor is not)
    const oq=below.map(o=>new THREE.Quaternion().copy(o.m.quaternion).invert()),os=below.map(o=>A.set(0,0,1).applyQuaternion(o.m.quaternion).y>=0?1:-1);
    for(let i=0;i<P.count;i++){rhP.set(B.getX(i),B.getY(i),0).applyQuaternion(q);const px=pos.x+rhP.x,pz=pos.z+rhP.z;let u=-1;
      for(const ox of [-.0107,0,.0107])for(const oz of [-.0107,0,.0107]){u=Math.max(u,fh(px+ox,pz+oz)+.0017);
        for(let j=0;j<below.length;j++){const o=below[j],op=o.m.position;rhL.set(px+ox-op.x,0,pz+oz-op.z).applyQuaternion(oq[j]);
          if(rhL.x<-PW/2||rhL.x>PW/2||rhL.y>0||rhL.y<-PH)continue;u=Math.max(u,op.y+os[j]*zOf(o.m.geometry,rhL.x,rhL.y)+.0019)}}
      drU[i]=u;drH[i]=u+.0012}
    // stiffness: relax toward the neighbours' average, never below what supports it
    for(let it=0;it<14;it++){for(let j=0;j<=PGY;j++)for(let i=0;i<=PGX;i++){const k=j*NX1+i;let a=0,n=0;if(i>0){a+=drH[k-1];n++}if(i<PGX){a+=drH[k+1];n++}if(j>0){a+=drH[k-NX1];n++}if(j<PGY){a+=drH[k+NX1];n++}drT[k]=Math.max(drU[k],.4*drH[k]+.6*a/n)}drH.set(drT)}
    // its own old cockle and creases on top, a little softened by lying flat
    // edges never lie quite flat: a hair of lift all round, and a corner or two that has curled up
    if(!f.cl){const rr=mulberry32(Math.floor(f.ph*1e4));f.cn=makeNoise(Math.floor(f.ph*997));f.cl=[0,1,2,3].map(()=>rr()<.45?.0015+Math.pow(rr(),1.5)*.007:rr()*.0012);f.cr=.05+rr()*.05}
    for(let i=0;i<P.count;i++){const lx=B.getX(i),ly=B.getY(i),ex=PW/2-Math.abs(lx),ey=Math.min(-ly,PH+ly),de=Math.min(ex,ey);
      let lift=(.0012+.0026*f.cn.n(lx*14+f.ph,ly*14))*Math.pow(Math.max(0,1-de/.024),2);
      const ci=(lx>0?1:0)+(ly<-PH/2?2:0),dc=Math.hypot(ex,ey);lift+=f.cl[ci]*Math.pow(Math.max(0,1-dc/f.cr),2);
      const hh=Math.max(drU[i]+.0002,drH[i]+B.getZ(i)*s*.8+lift);
      // where it runs into a wall it bends up against it instead of going through
      rhP.set(lx,ly,0).applyQuaternion(q);let wx=pos.x+rhP.x,wz=pos.z+rhP.z,wy=hh,pen=0;
      const lim=[[X0+.004,1,0],[X1-.004,-1,0],[Z0+.004,0,1],[Z1-.004,0,-1]];
      for(const [w,sx,sz] of lim){const d=sx?(w-wx)*sx:(w-wz)*sz;if(d>0){if(sx)wx=w;else wz=w;pen=Math.max(pen,d)}}
      if(pen>0){wy+=pen*.96+.002;rhL.set(wx-pos.x,wy-pos.y,wz-pos.z).applyQuaternion(drQ.copy(q).invert());P.setXYZ(i,rhL.x,rhL.y,rhL.z)}
      else P.setXYZ(i,lx,ly,s*(hh-pos.y))}
    P.needsUpdate=true;g.computeVertexNormals();g.computeBoundingSphere();f.draped=true;
    // anything that lies on top of it has to settle again
    for(const o of flyers)if(o!==f&&o.rest&&o.draped&&(o.restOrder||0)>(f.restOrder||0)&&Math.abs(o.m.position.x-pos.x)<.45&&Math.abs(o.m.position.z-pos.z)<.45)o.draped=false}
  function resetFloor(){flyers.forEach(f=>{if(!f.wall){f.rest=false;f.active=false}});
    floorInit.forEach(({f},i)=>{f.restOrder=i+1;f.draped=false});
    for(const {f,x,z,ry} of floorInit){f.m.geometry=flatVariants[f.v];E.set(-Math.PI/2-.012,ry,0,'YXZ');f.m.quaternion.setFromEuler(E);f.target.copy(f.m.quaternion);f.restY=restHeight(f,x,z,f.m.quaternion);f.active=true;f.rest=true;f.m.position.set(x,f.restY,z);
    f.vel.set(0,0,0);f.ang.set(0,0,0);f.rest=true;f.active=true;f.m.visible=true}}
  function windAt(p,out){
    const dx=p.x-WC.x,dy=(p.y-WC.y)*.8,dz=p.z-WC.z,fall=1/(1+(dx*dx+dy*dy+dz*dz)*.35),w=S.interior*(.35+.65*sstep(0,1.2,p.y));
    const k=w*2.6*fall+w*.25;
    out.set(-k,.08*k,.28*k);
    out.x+=Math.sin(p.z*2.1+S.time*1.3)*.35*w;out.y=out.y*sstep(.05,.6,p.y)+Math.sin(p.x*1.7+S.time*1.1+p.z)*.3*w*sstep(.15,1.1,p.y);out.z+=Math.cos(p.x*2.3-S.time*.9)*.45*w;
    return out}
  function detach(s,kick=1){
    if(s.hidden)return;s.hidden=true;
    let f=flyers.find(q=>q.wall&&!q.active);if(!f){f=makeFlyer(s.v);f.wall=true}
    f.m.geometry=variants[s.v];f.v=s.v;s.m.decompose(f.m.position,f.m.quaternion,A);f.m.position.z+=.004;
    f.vel.set((simRand()-.5)*.2*kick,-.1-.2*simRand(),.35+.35*simRand()*kick);f.ang.set((simRand()-.5)*2.5*kick,(simRand()-.5)*1.5*kick,(simRand()-.5)*2*kick);
    f.rest=false;f.active=true;f.m.visible=true;f.restY=.004+simRand()*.01;updateCover()}
  function stepFlyers(dt){drapeBudget=6;
    for(const f of flyers){if(!f.active)continue;const p=f.m.position;windAt(p,W);const wm=W.length();
      if(f.rest){
        if(wm>.5){const k=(wm-.5);f.vel.x+=W.x*k*dt*2.5;f.vel.z+=W.z*k*dt*2.5;if(wm>2.1&&simRand()<dt*1.5){f.vel.y=.5+simRand()*.6*k;f.rest=false;f.draped=false;f.ang.set((simRand()-.5)*6,(simRand()-.5)*4,(simRand()-.5)*6)}}
        f.vel.x*=Math.exp(-dt*2.6);f.vel.z*=Math.exp(-dt*2.6);p.x=THREE.MathUtils.clamp(p.x+f.vel.x*dt,X0+.17,X1-.17);p.z=THREE.MathUtils.clamp(p.z+f.vel.z*dt,Z0+.04,Z1-.2);
        { let dx=0,dz=0;for(const sx of [-PW/2,PW/2])for(const sy of [0,-PH]){A.set(sx,sy,0).applyQuaternion(f.target);const cx=p.x+A.x,cz=p.z+A.z;
            if(cx<X0+.006)dx=Math.max(dx,X0+.006-cx);if(cx>X1-.006)dx=Math.min(dx,X1-.006-cx);if(cz<Z0+.006)dz=Math.max(dz,Z0+.006-cz);if(cz>Z1-.006)dz=Math.min(dz,Z1-.006-cz)}
          if(dx||dz){p.x+=dx;p.z+=dz;f.vel.x*=.3;f.vel.z*=.3;f.settle=1}}
        if((Math.abs(f.vel.x)+Math.abs(f.vel.z)>.004||f.settle>=1)&&(f.settle=(f.settle||0)+dt)>.1){f.settle=0;f.restY=restHeight(f,p.x,p.z,f.target)}
        p.y+=(f.restY-p.y)*Math.min(1,dt*9);
        const sp=Math.hypot(f.vel.x,f.vel.z);if(sp>.01){Q.setFromAxisAngle(A.set(0,1,0),sp*dt*1.5*(f.ph>3?1:-1));f.target.premultiply(Q)}
        f.m.quaternion.slerp(f.target,Math.min(1,dt*5));f.m.material=f.tone||(f.tone=(()=>{const m=pmRest.clone(),k=.8+.2*((f.ph*13.7)%1),w=(f.ph*7.3)%1;m.color.setRGB(k*(1+.02*w),k,k*(.97-.03*w));return m})());
        if(f.dpos&&(Math.hypot(p.x-f.dpos.x,p.z-f.dpos.z)>.005||Math.abs(p.y-f.dpos.y)>.0008||f.m.quaternion.angleTo(f.dq)>.015))f.draped=false;
        if(!f.draped&&drapeBudget-->0){drape(f);(f.dpos||(f.dpos=new THREE.Vector3())).copy(p);(f.dq||(f.dq=new THREE.Quaternion())).copy(f.m.quaternion)}
        f.m.geometry=f.dg&&f.dgv===f.v?f.dg:flatVariants[f.v];
      }else{f.m.material=pm;f.m.geometry=variants[f.v];
        // a falling sheet: air resists it mostly across its face, the pressure acts ahead of its middle so it flips, glides and swings down
        const n=FN.set(0,0,1).applyQuaternion(f.m.quaternion);const rel=FR.copy(f.vel).sub(W),vn=rel.dot(n);
        f.vel.addScaledVector(n,-(13*vn*Math.abs(vn)+1.1*vn)*dt);FT.copy(rel).addScaledVector(n,-vn);f.vel.addScaledVector(FT,-.8*dt);f.vel.y-=9.8*dt;
        FA.crossVectors(n,FT).multiplyScalar(vn*7);f.ang.addScaledVector(FA,dt);f.ang.x+=Math.sin(S.time*5.3+f.ph)*(.6+wm)*dt;f.ang.z+=Math.cos(S.time*4.1+f.ph*2)*(.6+wm)*dt;f.ang.multiplyScalar(Math.exp(-dt*1.6));
        const a=f.ang.length();if(a>1e-4){Q.setFromAxisAngle(A.copy(f.ang).multiplyScalar(1/a),a*dt);f.m.quaternion.premultiply(Q)}
        p.addScaledVector(f.vel,dt);if(p.z<Z0+.09&&p.y>.05){p.z=Z0+.09;f.vel.z=Math.abs(f.vel.z)*.3}
        if(p.x<X0+.08){p.x=X0+.08;f.vel.x*=-.3}if(p.x>X1-.08&&!(p.z>WIN.z0&&p.z<WIN.z1&&p.y>WIN.y0&&p.y<WIN.y1)){p.x=X1-.08;f.vel.x*=-.3}
        if(p.z<Z0+.08){p.z=Z0+.08;f.vel.z*=-.3}if(p.z>Z1-.08){p.z=Z1-.08;f.vel.z*=-.3}if(p.y>RH-.1){p.y=RH-.1;f.vel.y*=-.3}
        if(f.vel.y<=0&&p.y<.25&&(f.lc=(f.lc||0)+1)%3===0){f.restOrder=1e9;flatTarget(f);f.landY=restHeight(f,p.x,p.z,f.target)}
        if(p.y<=Math.max((floorH?floorH(p.x,p.z):0)+.025,(f.landY||0)+.012)&&f.vel.y<=0){flatTarget(f);f.restOrder=1e9;f.restY=restHeight(f,p.x,p.z,f.target);p.y=Math.max(p.y,f.restY);f.vel.y=0;f.vel.x*=.5;f.vel.z*=.5;f.rest=true;f.settle=0;f.restOrder=++restCounter;f.draped=false}
      }}}


  /* bare bulb on a cord, swung by the draught */
  const LA=new THREE.Vector3(-.3,RH-.24,-.55),LL=.95;
  const lampG=new THREE.Group();scene.add(lampG);
  const blackM=new THREE.MeshStandardMaterial({color:'#1d1b18',roughness:.55});
  const rose=new THREE.Mesh(new THREE.CylinderGeometry(.035,.04,.02,24),new THREE.MeshStandardMaterial({color:'#d9d6cc',roughness:.8}));rose.position.copy(LA).y-=.01;scene.add(rose);
  const cord=new THREE.Mesh(new THREE.CylinderGeometry(.003,.003,1,6),blackM);cord.castShadow=true;scene.add(cord);
  const socket=new THREE.Mesh(new THREE.CylinderGeometry(.017,.019,.055,20),blackM);lampG.add(socket);socket.position.y=.0275;
  const bulbM=new THREE.MeshStandardMaterial({color:'#d9d6cf',emissive:'#ffd59c',emissiveIntensity:0,roughness:.08,metalness:.1,transparent:true,opacity:.82});
  // the scanned bulb: clear glass (with its printed label and smudges), brass cap, filament on its glass stem
  let bulbBase=null,bulbGlass=null;
  if(TX.bulbObj){let src=null;TX.bulbObj.traverse(o=>{if(o.isMesh&&!src)src=o});
    const g=src.geometry,mats=Array.isArray(src.material)?src.material:[src.material],groups=g.groups.length?g.groups:[{start:0,count:g.attributes.position.count,materialIndex:0}];
    const part=gr=>{const ng=new THREE.BufferGeometry();for(const k of ['position','uv']){const a=g.attributes[k];if(a)ng.setAttribute(k,new THREE.BufferAttribute(a.array.slice(gr.start*a.itemSize,(gr.start+gr.count)*a.itemSize),a.itemSize))}
      const m=mergeVertices(ng,1e-6);m.computeVertexNormals();return m};
    const baseM=new THREE.MeshStandardMaterial({color:'#ffffff',roughness:1,metalness:1,emissive:'#ffb46a',emissiveIntensity:0});
    if(TX.buD){baseM.map=TX.buD;baseM.normalMap=TX.buN;baseM.roughnessMap=TX.buR;baseM.metalnessMap=TX.buM;baseM.emissiveMap=TX.buE}
    const glassM=new THREE.MeshPhysicalMaterial({color:'#ffffff',roughness:.06,metalness:0,ior:1.5,transparent:true,opacity:.55,depthWrite:false,side:THREE.DoubleSide,specularIntensity:1,clearcoat:.4,emissive:'#ffcf96',emissiveIntensity:0});
    if(TX.buD){glassM.map=TX.buD;glassM.alphaMap=TX.buA;glassM.roughnessMap=TX.buR}
    const holder=new THREE.Group();holder.rotation.x=Math.PI;holder.position.y=.008;lampG.add(holder);
    for(const gr of groups){const nm=(mats[gr.materialIndex]||mats[0]).name;const isGlass=/glass/i.test(nm);
      const m=new THREE.Mesh(part(gr),isGlass?glassM:baseM);m.castShadow=!isGlass;m.renderOrder=isGlass?3:0;holder.add(m);if(isGlass)bulbGlass=m;else bulbBase=m}}
  if(!bulbBase){const b=new THREE.Mesh(new THREE.SphereGeometry(.03,24,16),bulbM);b.position.y=-.045;lampG.add(b)}
  const bulbL=new THREE.PointLight('#ffcf94',15,0,2);bulbL.position.y=-.034;bulbL.userData.ptKeep=true;lampG.add(bulbL);
  const hg=new THREE.BufferGeometry();hg.setAttribute('position',new THREE.Float32BufferAttribute([0,0,0],3));
  const halo=new THREE.Points(hg,new THREE.PointsMaterial({size:.32,map:tex(dotSprite(),{srgb:true}),color:'#ffd9a8',transparent:true,opacity:.55,depthWrite:false,blending:THREE.AdditiveBlending}));halo.position.y=-.036;halo.frustumCulled=false;lampG.add(halo);
  bulbL.castShadow=true;bulbL.visible=false;bulbL.shadow.radius=4;{const sm=Math.min(screen.width,screen.height)<700&&!STILL;bulbL.shadow.mapSize.set(sm?512:1024,sm?512:1024)}bulbL.shadow.bias=-.002;bulbL.shadow.normalBias=.02;bulbL.shadow.camera.near=.05;bulbL.shadow.camera.far=8;
  const lamp={a1:.05,a2:-.03,w1:0,w2:0,on:false};
  const bob=new THREE.Vector3(),UP=new THREE.Vector3(0,1,0),LW=new THREE.Vector3();
  function placeLamp(){
    bob.set(LL*Math.sin(lamp.a1),-LL*Math.cos(lamp.a1)*Math.cos(lamp.a2),LL*Math.sin(lamp.a2)).add(LA);
    lampG.position.copy(bob);lampG.quaternion.setFromUnitVectors(UP,LW.copy(LA).sub(bob).normalize());
    cord.position.copy(LA).add(bob).multiplyScalar(.5);cord.scale.set(1,LA.distanceTo(bob)-.05,1);cord.quaternion.copy(lampG.quaternion);
  }
  function stepLamp(dt){
    windAt(bob,LW);const g=9.8/LL,kw=1.4/LL,c=.22;
    const n=Math.min(4,Math.ceil(dt/.008)),h=dt/n;
    for(let i=0;i<n;i++){
      lamp.w1+=(-g*Math.sin(lamp.a1)-c*lamp.w1+kw*LW.x*Math.cos(lamp.a1))*h;
      lamp.w2+=(-g*Math.sin(lamp.a2)-c*lamp.w2+kw*LW.z*Math.cos(lamp.a2))*h;
      lamp.a1+=lamp.w1*h;lamp.a2+=lamp.w2*h}
    placeLamp();
    const fl=lamp.on?1-.06*Math.max(0,Math.sin(S.time*31)*Math.sin(S.time*7.3))*Math.min(1,S.interior):0;
    bulbL.intensity=15*fl;halo.visible=lamp.on;halo.material.opacity=.4*fl;bulbM.emissiveIntensity=lamp.on?6*fl:0;bulbM.color.set(lamp.on?'#fff6e6':'#cfcac0');
    if(bulbBase){bulbBase.material.emissiveIntensity=lamp.on?60*fl:0;bulbGlass.material.emissiveIntensity=lamp.on?.9*fl:0}
  }
  function setLamp(on){lamp.on=on;bulbL.visible=on;bulbM.opacity=on?1:.82;stepLamp(0)}
  window.__lampPos=()=>lampG.getWorldPosition(new THREE.Vector3()).toArray().map(v=>+v.toFixed(3));
  placeLamp();

  /* ---------- story: white chair, paper roll on zip ties, stone wrapped in paper ---------- */
  const Yax=new THREE.Vector3(0,1,0),UPV=Yax;
  const chairPos=new THREE.Vector3(-1.25,0,2.05);
  const fdir=new THREE.Vector3(1.55-chairPos.x,0,-2.25-chairPos.z).normalize();
  const yaw0=Math.atan2(fdir.x,fdir.z),Q0=new THREE.Quaternion().setFromAxisAngle(Yax,yaw0);
  const M0=new THREE.Matrix4().compose(chairPos,Q0,new THREE.Vector3(1,1,1));
  const chairRight=new THREE.Vector3(1,0,0).applyQuaternion(Q0);
  const chairM=new THREE.MeshStandardMaterial({color:'#e9e5da',roughness:.36,metalness:0});
  if(TX.chD){chairM.map=TX.chD;chairM.normalMap=TX.chN;chairM.roughnessMap=TX.chR;chairM.roughness=1;chairM.color.set('#e6e2d8')}
  const chairG=new THREE.Group();chairG.matrixAutoUpdate=false;scene.add(chairG);
  // mass properties of the plastic shell, straight from its triangles
  const CH_MASS=2.4,comL=new THREE.Vector3(),IbC=new THREE.Matrix3();
  { const tris=[];let area=0;const a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3(),ab=new THREE.Vector3(),ac=new THREE.Vector3();
    if(TX.chairObj)TX.chairObj.traverse(o=>{if(!o.isMesh)return;const m=new THREE.Mesh(o.geometry,chairM);m.castShadow=true;m.receiveShadow=true;chairG.add(m);
      const p=o.geometry.attributes.position;for(let i=0;i+2<p.count;i+=3){a.fromBufferAttribute(p,i);b.fromBufferAttribute(p,i+1);c.fromBufferAttribute(p,i+2);
        const ar=ab.subVectors(b,a).cross(ac.subVectors(c,a)).length()/2;if(ar<1e-9)continue;tris.push([(a.x+b.x+c.x)/3,(a.y+b.y+c.y)/3,(a.z+b.z+c.z)/3,ar]);area+=ar}});
    for(const t of tris){comL.x+=t[0]*t[3];comL.y+=t[1]*t[3];comL.z+=t[2]*t[3]}comL.multiplyScalar(1/area);
    let xx=0,yy=0,zz=0,xy=0,xz=0,yz=0;for(const t of tris){const m=CH_MASS*t[3]/area,x=t[0]-comL.x,y=t[1]-comL.y,z=t[2]-comL.z;xx+=m*(y*y+z*z);yy+=m*(x*x+z*z);zz+=m*(x*x+y*y);xy-=m*x*y;xz-=m*x*z;yz-=m*y*z}
    IbC.set(xx,xy,xz,xy,yy,yz,xz,yz,zz)}
  // zip ties around the top rail of the back
  const tieM=new THREE.MeshStandardMaterial({color:'#121212',roughness:.45,polygonOffset:true,polygonOffsetFactor:-2,polygonOffsetUnits:-4});
  const TIEX=.088;   // right over the two narrow slots in the back, just under the top edge
  const TIEMESH=[];
  for(const x of [-TIEX,TIEX]){const t=new THREE.Mesh(new THREE.TorusGeometry(.024,.0032,6,28),tieM);t.rotation.y=Math.PI/2;t.position.set(x,.862,-.318);t.castShadow=true;chairG.add(t);
    const tail=new THREE.Mesh(new THREE.BoxGeometry(.0045,.0025,.035),tieM);tail.position.set(x,.877,-.352);tail.rotation.x=.75;tail.castShadow=true;chairG.add(tail);
    const head=new THREE.Mesh(new THREE.BoxGeometry(.009,.008,.008),tieM);head.position.set(x,.884,-.334);head.castShadow=true;chairG.add(head);TIEMESH.push({t,tail,head,x})}

  /* distance field of the chair (baked from the mesh): lets paper, stone and roll touch its real surface */
  const SDF={lo:[-.42,-.04,-.44],h:.007,n:[121,147,122],cols:12,d:null,W:0};
  if(TX.sdf&&TX.sdf.image){const im=TX.sdf.image,c=canvas(im.width,im.height),x=c.getContext('2d');x.drawImage(im,0,0);
    const dd=x.getImageData(0,0,c.width,c.height).data;SDF.W=c.width;SDF.d=new Uint8Array(c.width*c.height);for(let i=0;i<SDF.d.length;i++)SDF.d[i]=dd[i*4]}
  let HF=null;if(TX.hf){HF=TX.hf;HF.sv=Float32Array.from(HF.seat.v);HF.bv=Float32Array.from(HF.back.v)}
  const bil=(arr,nx,ny,fx,fy)=>{fx=Math.min(nx-1.001,Math.max(0,fx));fy=Math.min(ny-1.001,Math.max(0,fy));const ix=fx|0,iy=fy|0,tx=fx-ix,ty=fy-iy,k=iy*nx+ix;
    return (arr[k]*(1-tx)+arr[k+1]*tx)*(1-ty)+(arr[k+nx]*(1-tx)+arr[k+nx+1]*tx)*ty};
  const seatH=(x,z)=>HF?bil(HF.sv,HF.seat.nx,HF.seat.nz,(x-HF.x0)/HF.h,(z-HF.seat.z0)/HF.h):.444;
  const backZ=(x,y)=>HF?bil(HF.bv,HF.back.nx,HF.back.ny,(x-HF.x0)/HF.h,(y-HF.back.y0)/HF.h):-.27;
  const sdfAt=(x,y,z,g)=>{   // chair-local point -> distance to the shell (m); g receives the gradient
    const fx=(x-SDF.lo[0])/SDF.h,fy=(y-SDF.lo[1])/SDF.h,fz=(z-SDF.lo[2])/SDF.h,n=SDF.n;
    if(!SDF.d||fx<0||fy<0||fz<0||fx>=n[0]-1||fy>=n[1]-1||fz>=n[2]-1){if(g)g.set(0,0,0);return .1275}
    const ix=fx|0,iy=fy|0,iz=fz|0,tx=fx-ix,ty=fy-iy,tz=fz-iz,W=SDF.W,d=SDF.d,cols=SDF.cols;
    const o0=(((iz/cols)|0)*n[1]+iy)*W+(iz%cols)*n[0]+ix,o1=((((iz+1)/cols)|0)*n[1]+iy)*W+((iz+1)%cols)*n[0]+ix;
    const a=d[o0],b=d[o0+1],c=d[o0+W],e=d[o0+W+1],f=d[o1],gg=d[o1+1],hh=d[o1+W],k=d[o1+W+1];
    const x00=a+(b-a)*tx,x10=c+(e-c)*tx,x01=f+(gg-f)*tx,x11=hh+(k-hh)*tx,y0=x00+(x10-x00)*ty,y1=x01+(x11-x01)*ty;
    if(g){const dx0=(b-a)+((e-c)-(b-a))*ty,dx1=(gg-f)+((k-hh)-(gg-f))*ty;g.set((dx0+(dx1-dx0)*tz),((x10-x00)+((x11-x01)-(x10-x00))*tz),(y1-y0)).multiplyScalar(.0005/SDF.h)}
    return (y0+(y1-y0)*tz)*.0005};

  /* ---------- rigid bodies: chair, stone, falling roll ---------- */
  const G=9.8;
  const mkBody=(m,Ib,com)=>{const b={m,im:m>0?1/m:0,iIb:Ib?Ib.clone().invert():new THREE.Matrix3().set(0,0,0,0,0,0,0,0,0),com:com||new THREE.Vector3(),
    p:new THREE.Vector3(),q:new THREE.Quaternion(),v:new THREE.Vector3(),w:new THREE.Vector3(),iIw:new THREE.Matrix3(),R:new THREE.Matrix3(),fixed:false};return b};
  const _m4=new THREE.Matrix4(),_m3=new THREE.Matrix3();
  function updIw(b){_m4.makeRotationFromQuaternion(b.q);b.R.setFromMatrix4(_m4);if(b.fixed){b.iIw.set(0,0,0,0,0,0,0,0,0);return}
    b.iIw.multiplyMatrices(b.R,b.iIb).multiply(_m3.copy(b.R).transpose())}
  const CH=mkBody(CH_MASS,IbC,comL.clone());
  const STM=.7,STB=mkBody(STM,new THREE.Matrix3().set(STM/5*(.039*.039+.046*.046),0,0,0,STM/5*(.071*.071+.046*.046),0,0,0,STM/5*(.071*.071+.039*.039)));
  const RLB=mkBody(1.6,null);   // the roll while it is in the air: a point mass, its spin is tracked separately
  const toWorld=(b,l,out)=>out.copy(l).sub(b.com).applyQuaternion(b.q).add(b.p);
  const _qi=new THREE.Quaternion();
  const toLocal=(b,w,out)=>out.copy(w).sub(b.p).applyQuaternion(_qi.copy(b.q).invert()).add(b.com);
  const CM=new THREE.Matrix4();
  const chairMat=(out)=>{out.makeRotationFromQuaternion(CH.q);out.setPosition(V1.copy(CH.com).negate().applyQuaternion(CH.q).add(CH.p));return out};
  const L=(x,y,z)=>new THREE.Vector3(x,y,z).applyMatrix4(CM);
  const cl=x=>Math.min(1,Math.max(0,x));
  const V1=new THREE.Vector3(),V2=new THREE.Vector3(),V3=new THREE.Vector3(),V4=new THREE.Vector3(),GR=new THREE.Vector3();
  // floor height (real plank relief) and its envelope
  const fh=(x,z)=>floorH?floorH(x,z):0;
  const fN=(x,z,out)=>{const e=.006;return out.set(-(fh(x+e,z)-fh(x-e,z))/(2*e),1,-(fh(x,z+e)-fh(x,z-e))/(2*e)).normalize()};
  const paperFloor=(x,z)=>Math.max(fh(x,z),fh(x+.012,z),fh(x-.012,z),fh(x,z+.012),fh(x,z-.012));
  // the chair stands on four plastic feet; the back-left leg is 3 mm short, so it rocks on its diagonal like a real one
  const LEGS=[[.3,.275],[-.3,.275],[.214,-.281],[-.214,-.281]];
  const FEET=[];LEGS.forEach(([x,z],i)=>{for(const [dx,dz] of [[.011,.011],[-.011,.011],[.011,-.011],[-.011,-.011]])FEET.push({l:new THREE.Vector3(x+dx,i===3?.003:0,z+dz),leg:i,jn:0,jt:new THREE.Vector3()})});
  const HULL=[[.3,.88,-.33],[-.3,.88,-.33],[.32,.66,.0],[-.32,.66,.0],[.3,.45,.29],[-.3,.45,.29],[0,.88,-.33]].map(a=>new THREE.Vector3(...a));
  const SEATY=.457,EDGE=new THREE.Vector3(0,.448,.288);   // seat top, and the rounded front lip the paper runs over

  // contacts, solved with sequential impulses (normal + Coulomb friction, accumulated and clamped)
  const CT=[];let nCT=0;
  const ctNew=()=>{if(nCT>=CT.length)CT.push({key:null,a:null,b:null,ra:new THREE.Vector3(),rb:new THREE.Vector3(),n:new THREE.Vector3(),jt:new THREE.Vector3(),pen:0,mu:.5,e:0,jn:0,bias:0,vn0:0,kind:0});const c=CT[nCT++];c.jn=0;c.jt.set(0,0,0);c.b=null;c.kind=0;c.key=null;return c};
  const velAt=(b,r,out)=>out.crossVectors(b.w,r).add(b.v);
  const _t1=new THREE.Vector3(),_t2=new THREE.Vector3(),_t3=new THREE.Vector3(),_va=new THREE.Vector3(),_vb=new THREE.Vector3(),_J=new THREE.Vector3();
  const kAlong=(b,r,n)=>{if(!b)return 0;_t1.crossVectors(r,n).applyMatrix3(b.iIw);_t2.crossVectors(_t1,r);return b.im+n.dot(_t2)};
  const impulse=(b,r,J,s)=>{if(!b||b.fixed)return;b.v.addScaledVector(J,b.im*s);_t3.crossVectors(r,J).applyMatrix3(b.iIw);b.w.addScaledVector(_t3,s)};
  function addContact(a,b,pt,n,pen,mu,e,kind,key){const c=ctNew();c.a=a;c.b=b;c.ra.subVectors(pt,a.p);if(b)c.rb.subVectors(pt,b.p);c.n.copy(n);c.pen=pen;c.mu=mu;c.e=e;c.kind=kind||0;
    velAt(a,c.ra,_va);if(b)_va.sub(velAt(b,c.rb,_vb));c.vn0=_va.dot(n);
    if(key){c.key=key;if(key.jn>0){c.jn=key.jn*.85;c.jt.copy(key.jt).multiplyScalar(.85);c.jt.addScaledVector(n,-c.jt.dot(n));_J.copy(n).multiplyScalar(c.jn).add(c.jt);impulse(a,c.ra,_J,1);impulse(b,c.rb,_J,-1)}}
    return c}
  function solveContacts(h,iters){
    for(let it=0;it<iters;it++)for(let i=0;i<nCT;i++){const c=CT[i],a=c.a,b=c.b;
      velAt(a,c.ra,_va);if(b)_va.sub(velAt(b,c.rb,_vb));const vn=_va.dot(c.n);
      const kn=kAlong(a,c.ra,c.n)+kAlong(b,c.rb,c.n);if(kn<1e-9)continue;
      const tgt=Math.max(c.vn0<-.35?-c.e*c.vn0:0,Math.min(.2,Math.max(0,c.pen-.0005)*.12/h));
      let dj=(tgt-vn)/kn;const j0=c.jn;c.jn=Math.max(0,j0+dj);dj=c.jn-j0;
      _J.copy(c.n).multiplyScalar(dj);impulse(a,c.ra,_J,1);impulse(b,c.rb,_J,-1);
      // friction
      velAt(a,c.ra,_va);if(b)_va.sub(velAt(b,c.rb,_vb));_va.addScaledVector(c.n,-_va.dot(c.n));const vt=_va.length();if(vt<1e-7)continue;
      _t2.copy(_va).multiplyScalar(1/vt);const kt=kAlong(a,c.ra,_t2)+kAlong(b,c.rb,_t2);if(kt<1e-9)continue;
      _J.copy(c.jt).addScaledVector(_t2,-vt/kt);const lim=c.mu*c.jn,jl=_J.length();if(jl>lim)_J.multiplyScalar(lim/jl);
      _t1.subVectors(_J,c.jt);c.jt.copy(_J);impulse(a,c.ra,_t1,1);impulse(b,c.rb,_t1,-1)}
    for(let i=0;i<nCT;i++){const c=CT[i];if(c.key){c.key.jn=c.jn;c.key.jt.copy(c.jt)}}}
  function integrate(b,h){if(b.fixed)return;b.p.addScaledVector(b.v,h);const wl=b.w.length();if(wl>1e-7){_qi.setFromAxisAngle(_t1.copy(b.w).multiplyScalar(1/wl),wl*h);b.q.premultiply(_qi).normalize()}}

  // ---- the stone ----
  // the stone (a real scanned rock, ~12.5 x 6 x 7.5 cm) as an ellipsoid for contacts; wrapped, the paper adds a few millimetres
  const SAX=new THREE.Vector3(.0625,.0306,.0378),SAXW=new THREE.Vector3(.0705,.039,.046),WRAPR=.05;
  const WINCROSS=new THREE.Vector3(GLX,WIN.y0+1.12,ZC+.317);
  const ST={hit:false,hitV:0,leg:-1,wrapped:true,broke:false,still:0,contact:false,active:false};
  const supportDir=(dx,dy,dz,out)=>{const S=ST.wrapped?SAXW:SAX;V2.set(dx,dy,dz).applyQuaternion(_qi.copy(STB.q).invert());const hx=S.x*S.x*V2.x,hy=S.y*S.y*V2.y,hz=S.z*S.z*V2.z,hh=Math.sqrt(hx*V2.x+hy*V2.y+hz*V2.z);
    return out.set(hx/hh,hy/hh,hz/hh).applyQuaternion(STB.q).add(STB.p)};
  const stoneSupport=(out)=>{V2.set(0,-1,0).applyQuaternion(_qi.copy(STB.q).invert());const S=ST.wrapped?SAXW:SAX;const hx=S.x*S.x*V2.x,hy=S.y*S.y*V2.y,hz=S.z*S.z*V2.z,hh=Math.sqrt(hx*V2.x+hy*V2.y+hz*V2.z);
    return out.set(hx/hh,hy/hh,hz/hh).applyQuaternion(STB.q).add(STB.p)};
  let aimPt,aimDir,aimN,SA;
  const stoneR=()=>ST.wrapped?.058:.05;
  function stoneContacts(onGlass){
    if(!ST.active)return;const p=STB.p;
    if(!ST.broke&&p.x<GLX+.03&&p.x>GLX-.3){ST.broke=true;STB.v.multiplyScalar(.86);onGlass&&onGlass()}
    stoneSupport(V3);const fl=fh(V3.x,V3.z),pen=fl-V3.y;const wasC=ST.contact;ST.contact=pen>-.0015;
    if(ST.contact&&!wasC&&ST.broke&&!ST.hit){const vy=-STB.v.y;if(vy>.35)window.dispatchEvent(new CustomEvent('stoneland',{detail:{v:vy}}))}
    if(pen>-.001){fN(V3.x,V3.z,V4);addContact(STB,null,V3,V4,pen,ST.wrapped?.55:.6,.28,1)}
    // walls: the ellipsoid's own extreme points
    supportDir(-1,0,0,V3);if(V3.x<X0)addContact(STB,null,V3,V4.set(1,0,0),X0-V3.x,.4,.3);
    supportDir(0,0,1,V3);if(V3.z>Z1)addContact(STB,null,V3,V4.set(0,0,-1),V3.z-Z1,.4,.3);
    supportDir(0,0,-1,V3);if(V3.z<Z0)addContact(STB,null,V3,V4.set(0,0,1),Z0-V3.z,.4,.3);
    // against the chair's real shape: three spheres along the stone's long axis
    const rs=ST.wrapped?.041:.033;V4.set(1,0,0).applyQuaternion(STB.q);
    for(const k of [-.03,0,.03]){V3.copy(p).addScaledVector(V4,k);toLocal(CH,V3,V1);const d=sdfAt(V1.x,V1.y,V1.z,GR);
      if(d<rs+.003&&GR.lengthSq()>1e-8){const n=_t3.copy(GR).normalize().applyQuaternion(CH.q).clone(),pt=V3.clone().addScaledVector(n,-rs);
        const c=addContact(STB,null,pt,n,rs-d,.3,.08,2);
        if(!ST.hit&&c.vn0<-.02){let best=0,bd=1e9;LEGS.forEach(([lx,lz],i)=>{const dd=Math.hypot(V1.x-lx,V1.z-lz);if(dd<bd){bd=dd;best=i}});
          if(V1.y<.35){ST.hit=true;ST.hitT=S.storyT;ST.hitV=-c.vn0;ST.damp=.12;ST.absorb=1;if(!CH.fixed){_J.copy(n).multiplyScalar(-STM*(-c.vn0)*1.08*.45);impulse(CH,V2.subVectors(pt,CH.p),_J,1)}ST.leg=best;if(!CH.fixed){S.shake=Math.max(S.shake,.06);window.dispatchEvent(new CustomEvent('knock'))}}}}
      V4.set(1,0,0).applyQuaternion(STB.q)}}
  function stoneForces(h){if(!ST.active)return;STB.v.y-=G*h;
    if(ST.contact){STB.w.multiplyScalar(Math.exp(-h*(ST.wrapped?1.1:2.6)));STB.v.x*=Math.exp(-h*.22);STB.v.z*=Math.exp(-h*.22);
      // the floor sags gently: before the hit it keeps the stone on its line, after the hit it leans it against the leg
      if(!ST.hit){const lo=(STB.p.x-aimPt.x)*aimN.x+(STB.p.z-aimPt.z)*aimN.z,lv=STB.v.x*aimN.x+STB.v.z*aimN.z,a=(-3.5*lo-2.2*lv)*h;STB.v.x+=aimN.x*a;STB.v.z+=aimN.z*a}
      else{const k=ST.damp>0?24:3.2;ST.damp-=h;STB.w.multiplyScalar(Math.exp(-h*(k+1)));STB.v.x*=Math.exp(-h*k);STB.v.z*=Math.exp(-h*k)}}}
  function launch(vh){const tw=1.6/vh,vy=(WINCROSS.y-SA.y+.5*G*tw*tw)/tw;
    STB.p.copy(SA);STB.v.set(aimDir.x*vh,vy,aimDir.z*vh);STB.q.identity();STB.w.copy(aimN).multiplyScalar(-9).add(V4.set(.8,1.2,-.6));Object.assign(ST,{kin:false,settle:null,hitT:0,wrapped:true,hit:false,hitV:0,leg:-1,broke:false,still:0,contact:false,active:true})}

  // ---- chair ----
  function chairContacts(){
    for(const f of FEET){toWorld(CH,f.l,V3);const pen=fh(V3.x,V3.z)-V3.y;if(pen>-.0015)addContact(CH,null,V3,UPV,pen,.45,.12,3,f);else{f.jn=0;f.jt.set(0,0,0)}}
    for(const hp of HULL){toWorld(CH,hp,V3);if(V3.x<X0+.005)addContact(CH,null,V3,V4.set(1,0,0),X0+.005-V3.x,.4,.1);if(V3.z>Z1-.005)addContact(CH,null,V3,V4.set(0,0,-1),V3.z-(Z1-.005),.4,.1);
      const pf=fh(V3.x,V3.z)-V3.y;if(pf>0)addContact(CH,null,V3,fN(V3.x,V3.z,V4),pf,.5,.1)}}
  // rest pose: drop it a hair above the planks and let it find its three (or four) legs
  CH.q.copy(Q0);CH.p.copy(comL).applyQuaternion(Q0).add(chairPos);CH.p.y+=.004;updIw(CH);
  for(let k=0;k<960;k++){const h=1/480;CH.v.y-=G*h;nCT=0;chairContacts();solveContacts(h,10);integrate(CH,h);updIw(CH);CH.v.multiplyScalar(.999);CH.w.multiplyScalar(.995)}
  const CH_REST={p:CH.p.clone(),q:CH.q.clone()};CH.v.set(0,0,0);CH.w.set(0,0,0);
  chairMat(CM);
  // throw: from outside, through the fortochka, toward the front-right leg
  function setAim(ax,az){aimPt=toWorld(CH,new THREE.Vector3(ax,0,az),new THREE.Vector3());aimPt.y=0;
    aimDir=new THREE.Vector3(aimPt.x-WINCROSS.x,0,aimPt.z-WINCROSS.z).normalize();aimN=new THREE.Vector3(-aimDir.z,0,aimDir.x);
    SA=WINCROSS.clone().addScaledVector(aimDir,-1.6);SA.y=WINCROSS.y+.62}
  setAim(.325,.28);
  let THROW=4.2,stoneVel=new THREE.Vector3(-1,0,0);const SEARCH=[];

  /* ---------- the roll ---------- */
  const RIBW=.36,R0=.055,RK=.62;   // width, outer radius, I/(m r^2) of a wound paper roll on a card core
  const ringC=canvas(128,128),rc=ringC.getContext('2d');rc.fillStyle='#e8e4da';rc.fillRect(0,0,128,128);
  rc.strokeStyle='rgba(120,112,96,.35)';for(let k=4;k<64;k+=2.2){rc.lineWidth=.8;rc.beginPath();rc.arc(64,64,k,0,7);rc.stroke()}
  rc.fillStyle='#6f6a60';rc.beginPath();rc.arc(64,64,14,0,7);rc.fill();rc.fillStyle='#b9b2a3';rc.beginPath();rc.arc(64,64,9,0,7);rc.fill();
  const capM=new THREE.MeshStandardMaterial({map:tex(ringC,{srgb:true}),roughness:.9});
  const roll=new THREE.Mesh(new THREE.CylinderGeometry(1,1,RIBW,48,1),[paperNorm(new THREE.MeshStandardMaterial({map:tex(tPaper,{srgb:true}),roughness:.85})),capM,capM]);
  roll.castShadow=true;roll.receiveShadow=true;scene.add(roll);
  const SLOPE=.02;   // the old floor sags toward the window corner
  const RS={ph:0,zr:.15,vz:0,ang:0,om:0,r:R0,U:0,head:new THREE.Vector3(),s:0,axis:new THREE.Vector3(),c:new THREE.Vector3(),exit:new THREE.Vector3(),landed:false,bump:0,hE:0,tension:0};
  const rollRadius=()=>Math.sqrt(Math.max(.0009,R0*R0-RS.U*.00042/Math.PI));
  const edgeW=new THREE.Vector3();
  function rollStep(h){const R=RS;
    if(R.ph===0)return;
    if(R.ph===1){   // rolling forward along the seat, laying paper behind it
      V1.set(0,0,1).applyQuaternion(CH.q);const slope=-V1.y;
      R.vz+=(.25+G*slope/(1+RK))*h;R.zr+=R.vz*h;R.U+=R.vz*h;R.r=rollRadius();R.ang+=R.vz/R.r*h;
      if(R.zr>EDGE.z-.006){R.ph=2;toWorld(CH,V1.set(0,seatH(0,.27)+2*PT+R.r,R.zr),RLB.p);velAt(CH,V2.subVectors(RLB.p,CH.p),RLB.v);RLB.v.addScaledVector(V3.set(0,0,1).applyQuaternion(CH.q),R.vz);RLB.v.y-=.05;R.om=R.vz/R.r}
      return}
    if(R.ph===2){   // in the air, unwinding like a yo-yo: the paper holds it back and tugs the seat edge
      toWorld(CH,EDGE,edgeW);V1.subVectors(edgeW,RLB.p);const dist=V1.length();V1.multiplyScalar(1/Math.max(dist,1e-4));
      const T=RLB.m*G*RK/(1+RK)*Math.max(0,V1.y);R.tension=T;
      RLB.v.addScaledVector(V1,T/RLB.m*h);if(!CH.fixed){_J.copy(V1).multiplyScalar(-T*h);impulse(CH,V2.subVectors(edgeW,CH.p),_J,1)}
      const dPrev=R.lastD||dist;if(dist>dPrev)R.U+=(dist-dPrev)*1.5;R.lastD=dist;
      const vAlong=-RLB.v.dot(V1);R.om=Math.max(R.om,vAlong/R.r);R.r=rollRadius();R.ang+=R.om*h;
      R.axis.set(1,0,0).applyQuaternion(CH.q);
      return}
    if(R.ph===3&&R.tug>0&&!CH.fixed){const k=R.tug/.45;R.tug-=h;toWorld(CH,EDGE,edgeW);_J.set(R.head.x,-.35,R.head.z).multiplyScalar(17*k*k*h);impulse(CH,V2.subVectors(edgeW,CH.p),_J,1)}
    if(R.ph===3){   // on the planks: skids while its spin and speed agree, then rolls down the sag, laying the strip
      const vc=R.s-R.om*R.r;
      if(Math.abs(vc)>.012){const f=-.35*G*Math.sign(vc);R.s+=f*h;R.om-=f/(RK*R.r)*h}
      else{const down=R.head.dot(fdir)*G*SLOPE,res=.05+.25*cl(1-Math.abs(R.s)/.06)+(R.c.z<Z0+.95?1.2*cl((Z0+.95-R.c.z)/.35):0)+(R.c.x>X1-.75?1.2*cl((R.c.x-(X1-.75))/.35):0);R.s+=(down/(1+RK)-res*Math.sign(R.s))*h;R.om=R.s/R.r}
      const np=V1.copy(R.c).addScaledVector(R.head,R.s*h);
      let hE=-1;for(const a of [-.6,-.3,0,.3,.6])for(const b of [-.15,0,.15]){hE=Math.max(hE,fh(np.x+R.head.x*a*R.r+R.axis.x*b,np.z+R.head.z*a*R.r+R.axis.z*b))}
      const d=hE-R.hE;if(d>.0004){R.s*=1-Math.min(.25,d*14);R.om*=1-Math.min(.25,d*14);R.bump+=Math.min(5,d*900);
        const ang=(simRand()-.5)*Math.min(.1,d*30);_qi.setFromAxisAngle(Yax,ang);R.head.applyQuaternion(_qi)}
      R.c.copy(np);R.c.y=hE+R.r+.0025;R.hE=hE;R.U+=Math.max(0,R.om*R.r*h);R.r=rollRadius();R.ang+=R.om*h;
      if(R.c.x>X1-.3||R.c.x<X0+.3||R.c.z>Z1-.3){R.s*=Math.exp(-h*6);R.om=R.s/R.r}
      R.axis.crossVectors(R.head,Yax).normalize().negate();
      if(Math.abs(R.s)<.004&&Math.abs(vc)<.02){R.ph=4;R.s=0;R.om=0}
      return}}
  function rollAirContacts(){if(RS.ph!==2)return;const r=RS.r;
    // the planks
    const fl=fh(RLB.p.x,RLB.p.z);if(RLB.p.y-r<fl+.0005){
      if(RLB.v.y<-.45){RLB.v.y=-RLB.v.y*.18;RLB.p.y=fl+r+.0005;RS.bump=6;S.shake=Math.max(S.shake,.04);window.dispatchEvent(new CustomEvent('knock',{detail:{roll:1}}))}
      else{const vh=V1.set(RLB.v.x,0,RLB.v.z);RS.head.copy(fdir);RS.s=vh.dot(RS.head);RS.c.copy(RLB.p);RS.c.y=fl+r+.0025;RS.hE=fl;RS.ph=3;RS.landed=true;RS.tension=0;RS.tug=.45;RS.axis.crossVectors(RS.head,Yax).normalize().negate()}}
    // the chair's lip and legs
    for(const k of [-.13,0,.13]){V3.copy(RLB.p).addScaledVector(RS.axis,k);toLocal(CH,V3,V1);const d=sdfAt(V1.x,V1.y,V1.z,GR);
      if(d<r+.002&&GR.lengthSq()>1e-8){V4.copy(GR).normalize().applyQuaternion(CH.q);V2.copy(V3).addScaledVector(V4,-r);addContact(RLB,CH,V2,V4,r-d,.4,.1,4)}}}

  /* ---------- the paper strip: a cloth-like sheet of particles that is stiff in bending and does not stretch ---------- */
  const NCOL=9,DS=.022,DC=RIBW/(NCOL-1),DSH=Math.hypot(DS,DC),MAXR=320,TIER=7,TIERF=11,TIEC=[2,6],PT=.0072;
  const NPART=MAXR*NCOL,PPOS=new Float32Array(NPART*3),POLD=new Float32Array(NPART*3),PIM=new Float32Array(NPART),PCF=new Uint8Array(NPART),PNRM=new Float32Array(NPART*3),PCN=new Float32Array(NPART*3);
  const SR={n:0,lastRest:DS,U0:0,frozenTo:-1};
  // holes torn by the zip ties: two at the top of the back, two at the back of the seat (alpha map over the first metre of the strip)
  const SHOW_TIES=false,HOLEL=1.2,holeC=canvas(256,768),hx=holeC.getContext('2d');const TIER2=1e9;
  function drawHoles(){hx.fillStyle='#fff';hx.fillRect(0,0,256,768);if(!SHOW_TIES)return;const hr=mulberry32(5);hx.fillStyle='#000';
    for(const row of [TIER,TIERF])for(const u of [TIEC[0]/(NCOL-1),TIEC[1]/(NCOL-1)]){
      // punched by hand with whatever was at hand: off-centre, ragged, a split running off it
      const cx=u*256+(hr()-.5)*3,cy=row*DS/HOLEL*768+(hr()-.5)*4,R0h=3.6+hr()*2.2,el=1.1+hr()*.8,rot=hr()*3.14;hx.beginPath();
      for(let k=0;k<=22;k++){const a=k/22*Math.PI*2,rr=R0h*(.7+.6*hr())*(1+.25*Math.sin(a*3+rot));const lx=Math.cos(a)*rr*el,ly=Math.sin(a)*rr;hx.lineTo(cx+lx*Math.cos(rot)-ly*Math.sin(rot),cy+lx*Math.sin(rot)+ly*Math.cos(rot))}hx.fill();
      hx.strokeStyle='#000';const nt=1+Math.floor(hr()*3);for(let k=0;k<nt;k++){let a=hr()*6.28,px=cx,py=cy;hx.lineWidth=1.4;hx.beginPath();hx.moveTo(px,py);const L=4+hr()*9;
        for(let s=0;s<5;s++){a+=(hr()-.5)*.9;px+=Math.cos(a)*L/5;py+=Math.sin(a)*L/5;hx.lineTo(px,py)}hx.stroke()}}}
  drawHoles();const holeT=new THREE.CanvasTexture(holeC);holeT.channel=1;holeT.wrapS=holeT.wrapT=THREE.ClampToEdgeWrapping;
  const ribM=paperNorm(new THREE.MeshStandardMaterial({map:tex(tPaper,{srgb:true}),color:'#fffaf1',alphaMap:holeT,alphaTest:.5,roughness:.85,side:THREE.DoubleSide}));{const cn=crumpleNormal(256,512,505,14,90,.17);cn.wrapS=cn.wrapT=THREE.RepeatWrapping;ribM.normalMap=cn;ribM.normalScale=new THREE.Vector2(1,1);ribM.polygonOffset=true;ribM.polygonOffsetFactor=-1;ribM.polygonOffsetUnits=-2}
  const SUBC=2*(NCOL-1)+1,MAXV=(2*MAXR-1)*SUBC;
  const ribGeo=new THREE.BufferGeometry(),rPos=new Float32Array(MAXV*3),rUv=new Float32Array(MAXV*2),rUv1=new Float32Array(MAXV*2),rIdx=new Uint32Array((2*MAXR-2)*(SUBC-1)*6);
  { let k=0;for(let i=0;i<2*MAXR-2;i++)for(let j=0;j<SUBC-1;j++){const A=i*SUBC+j,B=A+1,Cn=A+SUBC,D=Cn+1;rIdx[k++]=A;rIdx[k++]=Cn;rIdx[k++]=B;rIdx[k++]=B;rIdx[k++]=Cn;rIdx[k++]=D}}
  ribGeo.setAttribute('position',new THREE.BufferAttribute(rPos,3));ribGeo.setAttribute('uv',new THREE.BufferAttribute(rUv,2));ribGeo.setAttribute('uv1',new THREE.BufferAttribute(rUv1,2));ribGeo.setIndex(new THREE.BufferAttribute(rIdx,1));
  const ribbon=new THREE.Mesh(ribGeo,ribM);ribbon.castShadow=true;ribbon.receiveShadow=true;ribbon.frustumCulled=false;scene.add(ribbon);
  const side=new THREE.Vector3();
  const TIELF=[null,null],TIEL=TIEC.map(c=>new THREE.Vector3((c-(NCOL-1)/2)*DC,.845,-.352)),TIEL2=TIEC.map(c=>new THREE.Vector3((c-(NCOL-1)/2)*DC,.45,-.12));
  // where the strip leaves the roll, and which way the roll's axis points
  function rollFrame(){const R=RS;
    if(R.ph<=1){toWorld(CH,V1.set(0,seatH(0,Math.min(.27,R.zr))+2*PT+R.r,R.zr),R.c);R.axis.set(1,0,0).applyQuaternion(CH.q);V2.set(0,1,0).applyQuaternion(CH.q);R.exit.copy(R.c).addScaledVector(V2,-R.r-.0005)}
    else if(R.ph===2){R.c.copy(RLB.p);const i=(SR.n-2)*NCOL+4;V1.set(PPOS[i*3]-R.c.x,PPOS[i*3+1]-R.c.y,PPOS[i*3+2]-R.c.z);V1.addScaledVector(R.axis,-V1.dot(R.axis));if(V1.lengthSq()<1e-8)V1.set(0,1,0);R.exit.copy(R.c).addScaledVector(V1.normalize(),R.r+PT*.4)}
    else{R.exit.copy(R.c);R.exit.y-=R.r-.0016;R.exit.addScaledVector(R.head,-.004)}}
  const setRow=(r,cx,cy,cz,ax,ay,az,pin)=>{for(let c=0;c<NCOL;c++){const i=r*NCOL+c,o=(c-(NCOL-1)/2)*DC;PPOS[i*3]=cx+ax*o;PPOS[i*3+1]=cy+ay*o;PPOS[i*3+2]=cz+az*o;POLD[i*3]=PPOS[i*3];POLD[i*3+1]=PPOS[i*3+1];POLD[i*3+2]=PPOS[i*3+2];PIM[i]=pin?0:1}};
  // zip ties: out of the front hole, tight over the top of the paper on the rail, in at the back hole, and back round under the rail through the slot
  // the plastic's own faces, found in its distance field: back and front face of the back panel at a height
  function panelZ(x,y){let zb=null,zf=null;for(let z=-.40;z<=-.24;z+=.0008){if(sdfAt(x,y,z)<.0019){if(zb===null)zb=z;zf=z}}return zb===null?null:[zb,zf]}
  // zip ties: a flat 4.5 mm strap, pulled tight. Out of the front hole, over the paper on the top edge, in at the back hole,
  // down behind the paper on the back of the panel, through the slot under the top edge, up the front and back out through the front hole
  function strap(pts,w,th){const curve=new THREE.CatmullRomCurve3(pts,false,'centripetal'),N=Math.max(8,pts.length*6),P=[],I=[];
    for(let s=0;s<=N;s++){const p=curve.getPoint(s/N),tg=curve.getTangent(s/N),side=new THREE.Vector3(1,0,0),nrm=new THREE.Vector3().crossVectors(tg,side).normalize();
      for(const [a,b] of [[-w,-th],[w,-th],[w,th],[-w,th]])P.push(p.x+side.x*a+nrm.x*b,p.y+side.y*a+nrm.y*b,p.z+side.z*a+nrm.z*b)}
    for(let s=0;s<N;s++)for(let q=0;q<4;q++){const a=s*4+q,b=s*4+(q+1)%4,c=a+4,d=b+4;I.push(a,c,b,b,c,d)}
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(P,3));g.setIndex(I);g.computeVertexNormals();return g}
  function buildTies(){
    if(!SHOW_TIES){for(const T of TIEMESH){T.t.visible=T.head.visible=T.tail.visible=false}return}
    for(let k=0;k<2;k++){const T=TIEMESH[k],x=T.x,pts=[];
      // over the top: on the paper, 1 mm proud of it
      for(let r=TIERF;r>=TIER;r--){const i=r*NCOL+TIEC[k];const L=toLocal(CH,V4.set(PPOS[i*3],PPOS[i*3+1],PPOS[i*3+2]),new THREE.Vector3());sdfAt(L.x,L.y,L.z,GR);
        const g=GR.lengthSq()>1e-10?GR.clone().normalize():new THREE.Vector3(0,1,0);L.addScaledVector(g,.0016);L.x=x;pts.push(L)}
      const F=pts[0],B=pts[pts.length-1];
      // the hidden run, hugging the plastic under the paper, through the slot
      const yS=.766,pB=panelZ(x,.80)||[-.30,-.296],pF=panelZ(x,F.y-.012)||[-.296,-.292],pB2=panelZ(x,B.y-.01)||[-.305,-.30];
      const back=[new THREE.Vector3(x,B.y-.008,pB2[0]-.0016),new THREE.Vector3(x,.80,pB[0]-.0016),new THREE.Vector3(x,yS+.006,pB[0]+.002)];
      const front=[new THREE.Vector3(x,yS+.006,pB[1]+.004),new THREE.Vector3(x,.80,pB[1]+.0016),new THREE.Vector3(x,F.y-.01,pF[1]+.0016)];
      T.t.geometry.dispose();T.t.geometry=mergeGeometries([strap(pts,.00225,.0006),strap([B].concat(back,front,[F]),.00225,.0006)]);T.t.position.set(0,0,0);T.t.rotation.set(0,0,0);T.t.scale.set(1,1,1);
      // the locking head sits on the front just above the hole; the cut end sticks out of it
      const p1=pts[1],tg=V3.subVectors(pts[2],pts[0]).normalize(),ang=Math.atan2(tg.y,tg.z);
      T.head.geometry.dispose();T.head.geometry=new THREE.BoxGeometry(.0068,.005,.008);T.head.position.copy(p1).add(V4.set(0,.0012,.0016));T.head.rotation.set(-ang,0,0);
      T.tail.geometry.dispose();T.tail.geometry=new THREE.BoxGeometry(.0036,.0009,.011);T.tail.position.set(p1.x+.0004,p1.y+.0045,p1.z+.006);T.tail.rotation.set(-ang+1.0,.12*(k?1:-1),0)}}
  /* the sheet over the back of the chair: for every column of the strip its own path round the backrest's real outline —
     the top of the backrest is arched and curved, so the paper follows it down at the sides; over the top edge it folds back
     and hangs behind in a loose accordion (a few soft pleats), pressed onto the plastic, never through it */
  const TAILL=[],COLP=[],TOPL=[],NPIN=10;
  { const sdA=(x,y,z)=>sdfAt(x,y,z,GR),cl=PT+.0012;
    const rearZ=(x,y)=>{for(let z=-.40;z<-.22;z+=.001)if(sdA(x,y,z)<.0015)return z;return null};
    const frontZ=(x,y)=>{for(let z=-.22;z>-.40;z-=.001)if(sdA(x,y,z)<.0015)return z;return null};
    const tn=makeNoise(313);
    for(let c=0;c<NCOL;c++){const x=(c-(NCOL-1)/2)*DC;
      // the highest point of the backrest at this x
      let yt0=0,zt=-.31;for(let z=-.37;z<-.26;z+=.002)for(let y=.95;y>.74;y-=.001)if(sdA(x,y,z)<.0015){if(y>yt0){yt0=y;zt=z}break}
      const ex=.003+.007*Math.max(0,(Math.abs(x)-.11)/.07);   // a little more air at the ends, where the top of the backrest turns down
      let zr=1e9;for(let y=yt0-.03;y<=yt0+.001;y+=.002){const z=rearZ(x,y);if(z!==null)zr=Math.min(zr,z)}if(zr>1)zr=zt-.015;zr-=cl+ex;
      const yc=yt0-.012;let zf=-1;for(let y=yc-.02;y<=yt0;y+=.002){const z=frontZ(x,y);if(z!==null)zf=Math.max(zf,z)}if(zf<-.5)zf=zt+.012;zf+=cl+ex;
      const zc=(zr+zf)/2,ac=(zf-zr)/2,b=yt0+cl+ex+.002-yc,P=[];
      // round the top edge, from behind to the front
      for(let i=0;i<=12;i++){const t=i/12*Math.PI;P.push(new THREE.Vector3(x,yc+b*Math.sin(t),zc-ac*Math.cos(t)))}
      // down the front of the backrest, close over the plastic
      let zl=zf;for(let y=yc-.025;y>.53;y-=.025){let z=HF&&y<.85&&Math.abs(x)<.235?backZ(x,y)+cl:(frontZ(x,y)??zl-cl)+cl;z=Math.max(z,zl-.002);zl=z;P.push(new THREE.Vector3(x,y,z))}
      P.push(new THREE.Vector3(x,.5,-.226),new THREE.Vector3(x,.468,-.196),new THREE.Vector3(x,SEATY+PT,-.14));
      COLP.push(P);
      // behind: from the top edge the paper falls back and hangs in a few pleats, gently drawn in toward the back of the panel
      const A=.011,dy=Math.sqrt(DS*DS-A*A)*.97;
      for(let k=1;k<=TIER;k++){const y=yc-k*dy,rz=rearZ(x,y),o=.003+(k%2?A:0)+(tn.n(c*.7+k*.9,k*1.3)-.5)*.004;
        const z=Math.min(rz===null?zr:rz-cl,zr+.22*(yc-y))-o;
        (TAILL[TIER-k]||(TAILL[TIER-k]=[]))[c]=new THREE.Vector3(x+(tn.n(c*.5+3,k*.8)-.5)*.004,y+(tn.n(c*.9,k*1.7+5)-.5)*.004,z)}}}
  function layStrip(){   // initial strip: the tail behind, round the top of the backrest, down its front, along the seat into the roll
    // the centre column decides how many rows there are; every column is walked the same distance along its own path
    const ext=c=>{const P=COLP[c],e=new THREE.Vector3((c-(NCOL-1)/2)*DC,SEATY+PT,RS.zr-.02);return P.concat([e])};
    const PX=[],LX=[];for(let c=0;c<NCOL;c++){const P=ext(c),a=[0];for(let k=1;k<P.length;k++)a.push(a[k-1]+P[k].distanceTo(P[k-1]));PX.push(P);LX.push(a)}
    const K=Math.floor(LX[(NCOL-1)/2][LX[(NCOL-1)/2].length-1]/DS)+1;let n=TIER+K;
    for(let c=0;c<NCOL;c++){const P=PX[c],a=LX[c];let j=1;
      for(let k=0;k<K;k++){const want=Math.min(k*DS,a[a.length-1]);while(j<a.length-1&&a[j]<want)j++;const t=(want-a[j-1])/Math.max(1e-6,a[j]-a[j-1]);
        V1.lerpVectors(P[j-1],P[j],Math.min(1,Math.max(0,t)));if(k<NPIN)(TOPL[k]||(TOPL[k]=[]))[c]=V1.clone();toWorld(CH,V1,V1);const i=((TIER+k)*NCOL+c)*3;PPOS[i]=POLD[i]=V1.x;PPOS[i+1]=POLD[i+1]=V1.y;PPOS[i+2]=POLD[i+2]=V1.z;PIM[(TIER+k)*NCOL+c]=1}}
    rollFrame();setRow(n,RS.exit.x,RS.exit.y,RS.exit.z,RS.axis.x,RS.axis.y,RS.axis.z,true);
    for(let r=0;r<TIER;r++)for(let c=0;c<NCOL;c++){const i=r*NCOL+c;V1.copy(TAILL[r][c]);toWorld(CH,V1,V1);PPOS[i*3]=POLD[i*3]=V1.x;PPOS[i*3+1]=POLD[i*3+1]=V1.y;PPOS[i*3+2]=POLD[i*3+2]=V1.z;PIM[i]=0}
    const j=(n-1)*NCOL+4;SR.n=n+1;SR.lastRest=Math.min(DS,Math.max(.002,Math.hypot(PPOS[j*3]-RS.exit.x,PPOS[j*3+1]-RS.exit.y,PPOS[j*3+2]-RS.exit.z)));SR.U0=RS.U;SR.frozenTo=0}
  // one constraint-projection pass helpers
  const dist=(i,j,rest,k)=>{const wi=PIM[i],wj=PIM[j],ws=wi+wj;if(ws===0)return;const a=i*3,b=j*3,dx=PPOS[b]-PPOS[a],dy=PPOS[b+1]-PPOS[a+1],dz=PPOS[b+2]-PPOS[a+2],d=Math.sqrt(dx*dx+dy*dy+dz*dz);if(d<1e-9)return;
    const s=k*(d-rest)/(d*ws);PPOS[a]+=dx*s*wi;PPOS[a+1]+=dy*s*wi;PPOS[a+2]+=dz*s*wi;PPOS[b]-=dx*s*wj;PPOS[b+1]-=dy*s*wj;PPOS[b+2]-=dz*s*wj};
  // bending with a real compliance (XPBD): the stiffness no longer depends on how many iterations run, so paper droops under its own weight
  const LBA=new Float32Array(NPART*3),LBC=new Float32Array(NPART*3);
  const bendX=(i,j,l,at,L,li,tx,ty,tz)=>{const wi=PIM[i],wj=PIM[j],wl=PIM[l],ws=wj+.25*(wi+wl);if(ws===0)return;const a=i*3,b=j*3,c=l*3;
    for(let k=0;k<3;k++){const e=PPOS[b+k]-.5*(PPOS[a+k]+PPOS[c+k])-(k===0?tx:k===1?ty:tz),dl=(-e-at*L[li+k])/(ws+at);L[li+k]+=dl;PPOS[b+k]+=wj*dl;PPOS[a+k]-=.5*wi*dl;PPOS[c+k]-=.5*wl*dl}};
  const bend=(i,j,l,k,tx,ty,tz)=>{const wi=PIM[i],wj=PIM[j],wl=PIM[l],ws=wj+.25*(wi+wl);if(ws===0)return;const a=i*3,b=j*3,c=l*3;
    const ex=PPOS[b]-.5*(PPOS[a]+PPOS[c])-tx,ey=PPOS[b+1]-.5*(PPOS[a+1]+PPOS[c+1])-ty,ez=PPOS[b+2]-.5*(PPOS[a+2]+PPOS[c+2])-tz,s=k/ws;
    PPOS[b]-=ex*s*wj;PPOS[b+1]-=ey*s*wj;PPOS[b+2]-=ez*s*wj;PPOS[a]+=ex*s*.5*wi;PPOS[a+1]+=ey*s*.5*wi;PPOS[a+2]+=ez*s*.5*wi;PPOS[c]+=ex*s*.5*wl;PPOS[c+1]+=ey*s*.5*wl;PPOS[c+2]+=ez*s*.5*wl};
  const CMi=new THREE.Matrix4(),CMp=new THREE.Matrix4();
  function collide(i,first){if(PIM[i]===0)return;const a=i*3;let x=PPOS[a],y=PPOS[a+1],z=PPOS[a+2],hit=0;
    // planks
    if(y<.025){const pf=paperFloor(x,z)+.0016;if(y<pf){PCN[a]=0;PCN[a+1]=pf-y;PCN[a+2]=0;y=pf;hit=1}}
    if(x<X0+.004){x=X0+.004}if(x>X1-.004)x=X1-.004;if(z<Z0+.004)z=Z0+.004;if(z>Z1-.004)z=Z1-.004;
    // chair shell
    const e=CMi.elements,lx=e[0]*x+e[4]*y+e[8]*z+e[12],ly=e[1]*x+e[5]*y+e[9]*z+e[13],lz=e[2]*x+e[6]*y+e[10]*z+e[14];
    if(lx>-.43&&lx<.43&&ly<.96&&lz>-.45&&lz<.41){
      // seat and back as closed surfaces (the slots in them are far too narrow for a sheet of paper)
      if(HF&&lx>-.235&&lx<.235){const m=CM.elements;
        if(lz>-.2&&lz<.272){const st=seatH(lx,lz)+PT;if(ly<st&&ly>st-PT-.0045){const pen=st-ly,wx=m[4]*pen,wy=m[5]*pen,wz=m[6]*pen;x+=wx;y+=wy;z+=wz;PCN[a]=wx;PCN[a+1]=wy;PCN[a+2]=wz;hit=2}}
        if(ly>.47&&ly<.85){const bz=backZ(lx,ly)+PT;if(lz<bz&&lz>bz-PT-.004){const pen=bz-lz,wx=m[8]*pen,wy=m[9]*pen,wz=m[10]*pen;x+=wx;y+=wy;z+=wz;PCN[a]=wx;PCN[a+1]=wy;PCN[a+2]=wz;hit=2}}}
      const d=sdfAt(lx,ly,lz,GR);
      if(d<PT){const gl=GR.length();if(gl>1e-6){const m=CM.elements,gx=GR.x/gl,gy=GR.y/gl,gz=GR.z/gl,wx=m[0]*gx+m[4]*gy+m[8]*gz,wy=m[1]*gx+m[5]*gy+m[9]*gz,wz=m[2]*gx+m[6]*gy+m[10]*gz,pen=PT-d;
        x+=wx*pen;y+=wy*pen;z+=wz*pen;PCN[a]=wx*pen;PCN[a+1]=wy*pen;PCN[a+2]=wz*pen;hit=2}}}
    // the roll
    { const r=RS.r+.0025,cx=x-RS.c.x,cy=y-RS.c.y,cz=z-RS.c.z,ad=cx*RS.axis.x+cy*RS.axis.y+cz*RS.axis.z;
      if(Math.abs(ad)<RIBW/2+.01&&(RS.ph===2||cy>-.5*RS.r)){const px=cx-RS.axis.x*ad,py=cy-RS.axis.y*ad,pz=cz-RS.axis.z*ad,pd=Math.sqrt(px*px+py*py+pz*pz);
        if(pd<r&&pd>1e-6&&((i/NCOL)|0)<SR.n-8){const s=(r-pd)/pd;x+=px*s;y+=py*s;z+=pz*s}}}
    // the stone
    if(ST.active||S.storyT>=0){const r=(ST.wrapped?.05:.042)+.004,dx=x-STB.p.x,dy=y-STB.p.y,dz=z-STB.p.z,dd=dx*dx+dy*dy+dz*dz;if(dd<r*r&&dd>1e-10){const d=Math.sqrt(dd),s=(r-d)/d;x+=dx*s;y+=dy*s;z+=dz*s}}
    PPOS[a]=x;PPOS[a+1]=y;PPOS[a+2]=z;if(hit)PCF[i]=hit}
  // chords between two particles may cut a sharp edge of the chair: push the pair out from the edge
  function collideChord(i,j){if(PIM[i]+PIM[j]===0)return;const a=i*3,b=j*3,x=.5*(PPOS[a]+PPOS[b]),y=.5*(PPOS[a+1]+PPOS[b+1]),z=.5*(PPOS[a+2]+PPOS[b+2]);
    const e=CMi.elements,lx=e[0]*x+e[4]*y+e[8]*z+e[12],ly=e[1]*x+e[5]*y+e[9]*z+e[13],lz=e[2]*x+e[6]*y+e[10]*z+e[14];
    if(!(lx>-.43&&lx<.43&&ly<.96&&lz>-.45&&lz<.41))return;const d=sdfAt(lx,ly,lz,GR);if(d>=PT*.8)return;const gl=GR.length();if(gl<1e-6)return;
    const m=CM.elements,gx=GR.x/gl,gy=GR.y/gl,gz=GR.z/gl,wx=m[0]*gx+m[4]*gy+m[8]*gz,wy=m[1]*gx+m[5]*gy+m[9]*gz,wz=m[2]*gx+m[6]*gy+m[10]*gz,pen=PT*.8-d,wi=PIM[i],wj=PIM[j],ws=wi+wj;
    PPOS[a]+=wx*pen*wi/ws*2;PPOS[a+1]+=wy*pen*wi/ws*2;PPOS[a+2]+=wz*pen*wi/ws*2;PPOS[b]+=wx*pen*wj/ws*2;PPOS[b+1]+=wy*pen*wj/ws*2;PPOS[b+2]+=wz*pen*wj/ws*2}
  const CURL=-DS*DS*.5/3.0;
  const BTA=new Float32Array(NPART),BTC=new Float32Array(NPART);
  { const cr=mulberry32(4242),cn=makeNoise(77);for(let r=0;r<MAXR;r++)for(let c=0;c<NCOL;c++){const i=r*NCOL+c,u=c/(NCOL-1),v=r*DS;
      // the tail behind the ties was scrunched when it was tied on: strong random creases, its end folded back under
      if(r<TIER){BTA[i]=0;BTC[i]=0}
      // pulled tight by the ties: the paper over the rail is creased and puckered toward each tie
      else if(r<=TIERF+4){const k=1-Math.min(Math.abs(c-TIEC[0]),Math.abs(c-TIEC[1]))/4;BTA[i]=(cn.n(u*5+1.7,v*30)-.5)*.007*(.5+k);BTC[i]=(cn.n(u*6+9,v*24)-.5)*.006*(.4+k)}
      // the rest keeps a faint memory of being handled: soft cockles
      else{BTA[i]=(cn.n(u*2.2+5,v*6)-.5)*.0012;BTC[i]=(cn.n(u*3+3,v*5+2)-.5)*.0009}}}   // a strip that lived on a roll wants to curl back, toward its inner face
  function stripStep(h){
    const n=SR.n,r0=0;if(n<2)return;
    chairMat(CM);CMi.copy(CM).invert();
    // pins: the zip ties and the roll
    for(let r=0;r<TIER;r++)for(let c=0;c<NCOL;c++){const i=r*NCOL+c;V1.copy(TAILL[r][c]).applyMatrix4(CM);PPOS[i*3]=V1.x;PPOS[i*3+1]=V1.y;PPOS[i*3+2]=V1.z;PIM[i]=0}
    for(let k=0;k<NPIN;k++)if(TOPL[k])for(let c=0;c<NCOL;c++){const i=(TIER+k)*NCOL+c;V1.copy(TOPL[k][c]).applyMatrix4(CM);PPOS[i*3]=V1.x;PPOS[i*3+1]=V1.y;PPOS[i*3+2]=V1.z;PIM[i]=0}
    rollFrame();for(let c=0;c<NCOL;c++){const i=(n-1)*NCOL+c,o=(c-(NCOL-1)/2)*DC;PPOS[i*3]=RS.exit.x+RS.axis.x*o;PPOS[i*3+1]=RS.exit.y+RS.axis.y*o;PPOS[i*3+2]=RS.exit.z+RS.axis.z*o;PIM[i]=0}
    // normals of the sheet (inner face of the old roll), for curl memory and the draught
    for(let r=Math.max(0,r0-1);r<n;r++){if(ROWF[r])continue;for(let c=0;c<NCOL;c++){const i=r*NCOL+c,ia=Math.min(n-1,r+1)*NCOL+c,ib=Math.max(0,r-1)*NCOL+c,ic=r*NCOL+Math.min(NCOL-1,c+1),id=r*NCOL+Math.max(0,c-1);
      const ax=PPOS[ia*3]-PPOS[ib*3],ay=PPOS[ia*3+1]-PPOS[ib*3+1],az=PPOS[ia*3+2]-PPOS[ib*3+2],bx=PPOS[ic*3]-PPOS[id*3],by=PPOS[ic*3+1]-PPOS[id*3+1],bz=PPOS[ic*3+2]-PPOS[id*3+2];
      let nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;const l=Math.hypot(nx,ny,nz)||1;PNRM[i*3]=nx/l;PNRM[i*3+1]=ny/l;PNRM[i*3+2]=nz/l}}
    // integrate
    const damp=Math.exp(-h*2.2),wk=S.interior;
    for(let r=r0;r<n;r++){if(ROWF[r])continue;for(let c=0;c<NCOL;c++){const i=r*NCOL+c;if(PIM[i]===0)continue;const a=i*3;
      let ax=0,ay=-G,az=0;if(wk>.05&&PPOS[a+1]>.02){windAt(V1.set(PPOS[a],PPOS[a+1],PPOS[a+2]),W);const wn=W.x*PNRM[a]+W.y*PNRM[a+1]+W.z*PNRM[a+2];ax+=PNRM[a]*wn*1.4;ay+=PNRM[a+1]*wn*1.4;az+=PNRM[a+2]*wn*1.4}
      for(let k=0;k<3;k++){const v=(PPOS[a+k]-POLD[a+k])*damp;POLD[a+k]=PPOS[a+k];PPOS[a+k]+=v+(k===0?ax:k===1?ay:az)*h*h}PCF[i]=0}}
    const its=5,atA=2.2e-4/(h*h),atC=1e-4/(h*h);LBA.fill(0,0,n*NCOL*3);LBC.fill(0,0,n*NCOL*3);
    for(let it=0;it<its;it++){
      for(let r=r0;r<n-1;r++){if(ROWF[r]&&ROWF[r+1])continue;const rest=r===n-2?SR.lastRest:DS;for(let c=0;c<NCOL;c++){const i=r*NCOL+c;dist(i,i+NCOL,rest,1)}}
      for(let r=Math.max(0,r0-1);r<n;r++){if(ROWF[r])continue;for(let c=0;c<NCOL-1;c++){const i=r*NCOL+c;dist(i,i+1,DC,1)}}
      for(let r=r0;r<n-2;r++){if(ROWF[r]&&ROWF[r+1])continue;for(let c=0;c<NCOL-1;c++){const i=r*NCOL+c;dist(i,i+NCOL+1,DSH,.5);dist(i+1,i+NCOL,DSH,.5)}}
      for(let r=Math.max(1,r0);r<n-2;r++){if(ROWF[r])continue;for(let c=0;c<NCOL;c++){const i=r*NCOL+c,a=i*3;{const t=CURL+BTA[i];bendX(i-NCOL,i,i+NCOL,atA,LBA,a,PNRM[a]*t,PNRM[a+1]*t,PNRM[a+2]*t)}}}
      for(let r=r0;r<n-1;r++){if(ROWF[r])continue;for(let c=1;c<NCOL-1;c++){const i=r*NCOL+c,a=i*3,t=BTC[i];bendX(i-1,i,i+1,atC,LBC,a,PNRM[a]*t,PNRM[a+1]*t,PNRM[a+2]*t)}}
      // never farther from the ties than the paper between them (long-range attachments keep it from stretching)
      for(let r=Math.max(r0,TIER+1);r<n-1;r++)for(let c=0;c<NCOL;c++){if(ROWF[r])break;const i=r*NCOL+c;if(PIM[i]===0)continue;const tr=r>TIER2?TIER2:TIER,t=tr*NCOL+(c<NCOL/2?TIEC[0]:TIEC[1]),ox=Math.abs(c-(c<NCOL/2?TIEC[0]:TIEC[1]))*DC,lim=Math.hypot((r-tr)*DS,ox)*1.002;
        const a=i*3,b=t*3,dx=PPOS[a]-PPOS[b],dy=PPOS[a+1]-PPOS[b+1],dz=PPOS[a+2]-PPOS[b+2],d=Math.sqrt(dx*dx+dy*dy+dz*dz);if(d>lim){const s=(lim-d)/d;PPOS[a]+=dx*s;PPOS[a+1]+=dy*s;PPOS[a+2]+=dz*s}}
      if(it===0||it===its-1||it===2){for(let r=r0;r<n;r++){if(ROWF[r])continue;for(let c=0;c<NCOL;c++)collide(r*NCOL+c,it===0)}
        if(it===its-1)for(let r=r0;r<n-1;r++){if(ROWF[r])continue;for(let c=0;c<NCOL;c+=2){const i=r*NCOL+c;collideChord(i,i+NCOL)}}}}
    // friction: paper sticks to planks and to the chair (which may be moving under it)
    for(let r=r0;r<n;r++)for(let c=0;c<NCOL;c++){const i=r*NCOL+c;if(!PCF[i]||PIM[i]===0)continue;const a=i*3;
      const nx=PCN[a],ny=PCN[a+1],nz=PCN[a+2],pn=Math.hypot(nx,ny,nz)||1e-9;
      let sx=0,sy=0,sz=0;if(PCF[i]===2){const e=CMi.elements,x=POLD[a],y=POLD[a+1],z=POLD[a+2];V1.set(e[0]*x+e[4]*y+e[8]*z+e[12],e[1]*x+e[5]*y+e[9]*z+e[13],e[2]*x+e[6]*y+e[10]*z+e[14]);V2.copy(V1).applyMatrix4(CM);V3.copy(V1).applyMatrix4(CMp);sx=V2.x-V3.x;sy=V2.y-V3.y;sz=V2.z-V3.z}
      let dx=PPOS[a]-POLD[a]-sx,dy=PPOS[a+1]-POLD[a+1]-sy,dz=PPOS[a+2]-POLD[a+2]-sz;const ux=nx/pn,uy=ny/pn,uz=nz/pn,dn=dx*ux+dy*uy+dz*uz;dx-=ux*dn;dy-=uy*dn;dz-=uz*dn;
      const dt=Math.hypot(dx,dy,dz),mu=PCF[i]===1?.8:.9;if(dt<1e-9)continue;const k=dt<mu*pn?1:mu*pn/dt;PPOS[a]-=dx*k;PPOS[a+1]-=dy*k;PPOS[a+2]-=dz*k}
    CMp.copy(CM);
    // the roll pays out paper: when the last segment is long enough it becomes part of the sheet
    SR.lastRest=Math.min(DS,SR.restAtSpawn+RS.U-SR.U0);
    if(SR.lastRest>=DS-1e-5&&n<MAXR){for(let c=0;c<NCOL;c++)PIM[(n-1)*NCOL+c]=1;rollFrame();setRow(n,RS.exit.x,RS.exit.y,RS.exit.z,RS.axis.x,RS.axis.y,RS.axis.z,true);SR.n=n+1;SR.U0=RS.U;SR.restAtSpawn=.0005;SR.lastRest=.0005}
  }
  // rows that have settled flat on the planks, away from the chair and the roll, are frozen
  const ROWF=new Uint8Array(MAXR);
  function freezeRows(){const n=SR.n;let hang=-1;
    for(let r=0;r<n-12;r++){if(ROWF[r])continue;const i=(r*NCOL+4)*3;if(PPOS[i+1]>paperFloor(PPOS[i],PPOS[i+2])+.012)hang=r}
    for(let r=hang+8;r<n-16;r++){if(ROWF[r])continue;let slow=true;for(let c=0;c<NCOL;c++){const i=(r*NCOL+c)*3;if(Math.abs(PPOS[i]-POLD[i])+Math.abs(PPOS[i+1]-POLD[i+1])+Math.abs(PPOS[i+2]-POLD[i+2])>.00012){slow=false;break}}
      if(slow){ROWF[r]=1;for(let c=0;c<NCOL;c++)PIM[r*NCOL+c]=0;SR.frozenTo++}}}
  // draw: Catmull-Rom through the particles, twice as fine in both directions
  const cr=(p0,p1,p2,p3,t)=>{const t2=t*t,t3=t2*t;return .5*((2*p1)+(-p0+p2)*t+(2*p0-5*p1+4*p2-p3)*t2+(-p0+3*p1-3*p2+p3)*t3)};
  const TMP=new Float32Array(MAXR*2*NCOL*3),OFS=new Float32Array(MAXV*3);
  // creases from being handled: different every time the room is set up
  const CREASE=[],PUCK=[];
  function makeCreases(seed){const cr=mulberry32(seed);CREASE.length=0;PUCK.length=0;
    for(let i=0;i<80;i++){const a=(cr()-.5)*2.8;CREASE.push({ca:Math.cos(a),sa:Math.sin(a),c:.05+cr()*(i<18?1.1:5.5),a:(cr()<.5?-1:1)*(.0007+cr()*.0015),w:.005+cr()*.009})}
    // where each tie bites, the paper is gathered into little radiating folds
    if(SHOW_TIES)for(let k=0;k<2;k++)for(let i=0;i<13;i++)PUCK.push({s:k?.5:-.5,a0:cr()*6.28,ang:(i/13)*Math.PI*2+cr()*.4,len:.025+cr()*.06,amp:(cr()<.5?-1:1)*(.0028+cr()*.004),w:.014+cr()*.02})}
  makeCreases(STILL?808:(Math.random()*1e6)|0);
  const CRK=(arc,s)=>{if(arc<.03)return 0;const e=s*s*s*s;
    let o=.0011*Math.sin(arc*13+1.7)*Math.sin(arc*3.1+s*.8)+.0016*e*(1+Math.sin(arc*21+s*3))+.0012*e;
    for(let k=0;k<CREASE.length;k++){const c=CREASE[k],d=arc*c.ca+s*.18*c.sa-c.c;if(d>-.04&&d<.04)o+=c.a*Math.exp(-(d*d)/(c.w*c.w))}
    const ta=(TIER+TIERF)*.5*DS;if(Math.abs(arc-ta)<.11){for(const p of PUCK){const dx=(s-p.s)*.18,dy=arc-ta,r=Math.hypot(dx,dy);if(r>p.len+.01||r<.004)continue;
      const an=Math.atan2(dy,dx),da=Math.atan2(Math.sin(an-p.ang),Math.cos(an-p.ang)),lat=Math.abs(Math.sin(da))*r;o+=p.amp*Math.max(0,1-lat/p.w)*Math.sin(Math.PI*Math.min(1,r/p.len))}}
    return o};
  window.__sd=(x,y,z)=>sdfAt(x,y,z,GR);
  window.__clipX=(x)=>{chairMat(CM);const n=new THREE.Vector3(-1,0,0).transformDirection(CM),p=new THREE.Vector3(x,0,0).applyMatrix4(CM);renderer.clippingPlanes=[new THREE.Plane().setFromNormalAndCoplanarPoint(n,p)]};
  window.__railCmp=()=>{const out=[];chairMat(CM);const rc=new THREE.Raycaster();chairG.updateMatrixWorld(true);
    for(const x of [-.12,-.06,0,.06,.12])for(const z of [-.34,-.33,-.32,-.31,-.30,-.29,-.28,-.27]){let ys=null;for(let y=.98;y>.7;y-=.001){if(sdfAt(x,y,z,GR)<.0015){ys=y;break}}
      const o=new THREE.Vector3(x,1.2,z).applyMatrix4(CM),dvec=new THREE.Vector3(0,-1,0).transformDirection(CM);rc.set(o,dvec);const h=rc.intersectObject(chairG,true).filter(q=>q.object.isMesh&&q.object.geometry.attributes.position.count>500);
      const ym=h.length?new THREE.Vector3().copy(h[0].point).applyMatrix4(CMi.copy(CM).invert()).y:null;out.push([x,z,ys&&+ys.toFixed(3),ym&&+ym.toFixed(3)])}return out};
  window.__stripPen=()=>{const n=SR.n,nr=2*n-1,out=[];chairMat(CM);CMi.copy(CM).invert();const me=CMi.elements;
    for(let i=0,m=nr*SUBC;i<m;i++){const o=i*3,x=rPos[o],y=rPos[o+1],z=rPos[o+2];const lx=me[0]*x+me[4]*y+me[8]*z+me[12],ly=me[1]*x+me[5]*y+me[9]*z+me[13],lz=me[2]*x+me[6]*y+me[10]*z+me[14];
      if(lx<-.43||lx>.43||ly<.02||lz<-.45||lz>.41)continue;const d=ly>.98?1:sdfAt(lx,ly,lz,GR);if(d<.0025)out.push([(i/SUBC)|0,i%SUBC,+lx.toFixed(3),+ly.toFixed(3),+lz.toFixed(3),+d.toFixed(4)])}return out};

  /* ---------- the top of the sheet, tied to the backrest ----------
     The sheet lies fixed over the top of the backrest; there it is drawn as its own fine mesh (attached to the chair) so that the two
     zip ties can really bite: each loops round the top of the backrest through the top of an outer slot, over the paper, pulled tight.
     Under a tie the paper is pressed onto the plastic and gathered in from both sides; the gathered paper stands up in folds that fan
     out from the tie and die away; between the ties it is drawn taut. Where a tie goes through the sheet, a small ragged hole. */
  const CAP={built:false,R0:0,mesh:null,ties:[]};
  function slotTop(x){const op=y=>{for(let z=-.37;z<-.23;z+=.002)if(sdfAt(x,y,z,GR)<.0016)return false;return true};for(let y=.83;y>.70;y-=.001){if(op(y)&&op(y-.002)&&op(y-.004))return y}return .785}
  function slotX(sg){let a=null,b=null;for(let x=.058;x<.12;x+=.001){const xx=sg*x;let open=true;for(let z=-.37;z<-.23;z+=.002){if(sdfAt(xx,.762,z,GR)<.0016){open=false;break}}if(open){if(a===null)a=x;b=x}else if(a!==null)break}return a===null?sg*.088:sg*(a+b)/2}
  function buildCap(){CAP.built=true;const R0=2*(TIER+NPIN-1),NR=R0+1,NC=SUBC,U=4;CAP.R0=R0;
    chairMat(CM);CMi.copy(CM).invert();
    // the drawn sheet there, in the chair's own frame
    const B=[];for(let r=0;r<NR;r++){const row=[];for(let j=0;j<NC;j++){const o=(r*SUBC+j)*3;row.push(new THREE.Vector3(rPos[o],rPos[o+1],rPos[o+2]).applyMatrix4(CMi))}B.push(row)}
    // four times finer, smooth (Catmull-Rom) both ways
    const up=(arr,n,get)=>{const out=[];for(let i=0;i<n-1;i++)for(let k=0;k<U;k++){const t=k/U,a=get(Math.max(0,i-1)),b=get(i),c=get(i+1),d=get(Math.min(n-1,i+2));out.push(new THREE.Vector3(cr(a.x,b.x,c.x,d.x,t),cr(a.y,b.y,c.y,d.y,t),cr(a.z,b.z,c.z,d.z,t)))}out.push(get(n-1).clone());return out};
    const rowsU=B.map(row=>up(row,NC,j=>row[j]));const NCu=rowsU[0].length;
    const G0=[];for(let j=0;j<NCu;j++){const col=up(null,NR,r=>rowsU[r][j]);for(let i=0;i<col.length;i++)(G0[i]||(G0[i]=[]))[j]=col[i]}
    const NRu=G0.length,dS=DS/2/U,SEND=(NRu-1)*dS;
    const _a=new THREE.Vector3(),_b=new THREE.Vector3();
    const at=(fi,xm)=>{fi=Math.min(NRu-1.0001,Math.max(0,fi));const i=Math.floor(fi),u=fi-i,f=Math.min(NCu-1.0001,Math.max(0,(xm/RIBW+.5)*(NCu-1))),j=Math.floor(f),t=f-j;
      _a.lerpVectors(G0[i][j],G0[i][j+1],t);_b.lerpVectors(G0[i+1][j],G0[i+1][j+1],t);return V2.lerpVectors(_a,_b,u)};
    // the two ties: at the outer slots, round the top of the backrest from the top of the slot at the back to the top of the slot in front
    const ties=[-1,1].map(sg=>{const x0=sg*.083,ys=.79,   // the two outer slots of the back, measured off the chair's shape
      j0=Math.round((x0/RIBW+.5)*(NCu-1));let iTop=0;for(let i=0;i<NRu;i++)if(G0[i][j0].y>G0[iTop][j0].y)iTop=i;
      let ir=0;for(let i=iTop;i>=0;i--)if(G0[i][j0].y<=ys+.004){ir=i;break}let jf=NRu-1;for(let i=iTop;i<NRu;i++)if(G0[i][j0].y<=ys+.004){jf=i;break}
      return {x0,ys,sr:ir*dS,sf:jf*dS,st:iTop*dS,seed:sg*1.7+3}});
    CAP.ties=ties;
    const tri=x=>Math.abs((x-Math.floor(x))-.5)*2;
    /* behind: the whole hanging accordion is caught by the two ties and pinched at each of them, like the twisted ends of a sweet wrapper.
       Every pleat runs into the pinch: between the ties the paper hangs out in a pouch whose folds are U-shaped arcs from one tie to the
       other, the lowest in the middle; outside the ties the folds fan out from the pinch to the free edges. Built in the paper's own
       coordinates (x across, h down from the top edge), so each fold is one line of the sheet carried from tie to tie. */
    const SB=ties[0].st-.012,XA=ties[0].x0,XB=ties[1].x0,ib=Math.round(SB/dS),RZC=new Map(),nn3=makeNoise(509),nn4=makeNoise(613),cr2=mulberry32(4242);
    // the back face of the backrest at (x,y): the middle of what is found a little either side (slots and odd bits ignored)
    const rz=(x,y)=>{const k=Math.round(x*500)+':'+Math.round(y*500);if(RZC.has(k))return RZC.get(k);const f=[];
      for(const ox of [-.03,-.015,0,.015,.03])for(let z=-.40;z<-.22;z+=.0015)if(sdfAt(x+ox,y,z,GR)<.0015){f.push(z);break}
      f.sort((a,b)=>a-b);const best=f.length?f[(f.length-1)>>1]:null;RZC.set(k,best);return best};
    // each tie runs down the back of the rail to the top of its slot: under it the whole bundle is pressed flat along that line
    for(const T of ties){T.yTop=at(ib,T.x0).y;T.gT=Math.max(.12,(T.yTop-T.ys-.003)/SB);T.sBack=Math.max(.006,SB-(T.yTop-T.ys-.006)/T.gT)}
    const gOf=x=>{const da=Math.abs(x-XA),db=Math.abs(x-XB),d=Math.min(da,db),inside=x>XA&&x<XB,gt=(da<db?ties[0]:ties[1]).gT;
      return inside?gt+(1-gt)*Math.pow(sstep(0,(XB-XA)/2,d),.7):gt+(.84-gt)*Math.pow(sstep(0,.1,d),.75)};
    // paper, not cloth: a set of folds at uneven spacing, each a sharp ridge or valley with flat faces either side, of its own depth,
    // running from pinch to pinch but not perfectly parallel, now and then dying out or breaking
    const FOLDS=[];{let u=.05+cr2()*.06;while(u<1.05){const sp=.075+cr2()*.13;FOLDS.push({u,w:sp*(.55+cr2()*.3),a:(cr2()<.78?1:-.7)*(.5+cr2()*1.1),o:cr2()*50});u+=sp}}
    const foldAt=(xm,u)=>{let f=0;for(const F of FOLDS){const uu=F.u+.035*(nn4.n(xm*5+F.o,F.o)-.5),t=Math.abs(u-uu)/F.w;if(t>=1)continue;
        const keep=sstep(.25,.45,nn4.n(xm*7+F.o*1.3,3.1+F.o));f+=F.a*keep*Math.pow(1-t,1.7)}return f};
    // the pouch is not a smooth bag either: flat panels meeting at a few bends
    const KX=[0,.22,.45,.63,.82,1].map((k,i)=>[k,i===0||i===5?0:Math.sin(Math.PI*k)*(.85+cr2()*.3)]),KU=[[0,0],[.18,.55+cr2()*.15],[.45,.95],[.72,.8+cr2()*.15],[1,.5]];
    const pl=(K,t)=>{t=Math.min(1,Math.max(0,t));for(let i=1;i<K.length;i++)if(t<=K[i][0]){const a=K[i-1],b=K[i];return a[1]+(b[1]-a[1])*(t-a[0])/(b[0]-a[0])}return K[K.length-1][1]};
    /* the paper hangs from the top edge, so it must clear everything of the chair that stands out behind above where it is: the curled-back
       rim along the top of the backrest and down its sides. For every x, the rearmost plastic at this height or anywhere above it */
    const EX0=-.21,EDX=.005,ENX=85,EY0=.97,EDY=.002,ENY=190,ENVT=new Float32Array(ENX*ENY);
    for(let ix=0;ix<ENX;ix++){const x=EX0+ix*EDX;let run=1;for(let iy=0;iy<ENY;iy++){const y=EY0-iy*EDY;
        for(let z=-.42;z<-.24;z+=.001)if(sdfAt(x,y,z,GR)<.0012){run=Math.min(run,z);break}ENVT[ix*ENY+iy]=run}}
    { const T=ENVT.slice();for(let ix=0;ix<ENX;ix++)for(let iy=0;iy<ENY;iy++){let m=1;for(let d=-2;d<=2;d++){const jx=ix+d;if(jx>=0&&jx<ENX)m=Math.min(m,T[jx*ENY+iy])}ENVT[ix*ENY+iy]=m}}
    const ENV=(x,y)=>{const fx=Math.min(ENX-1.001,Math.max(0,(x-EX0)/EDX)),fy=Math.min(ENY-1.001,Math.max(0,(EY0-y)/EDY)),ix=Math.floor(fx),iy=Math.floor(fy),tx=fx-ix,ty=fy-iy;
      const v=(a,b)=>{const q=ENVT[a*ENY+b];return q>0?-.2:q};return (v(ix,iy)*(1-tx)+v(ix+1,iy)*tx)*(1-ty)+(v(ix,iy+1)*(1-tx)+v(ix+1,iy+1)*tx)*ty};
    const candy=(xm,s,out)=>{const h=SB-s,u=h/SB,g=gOf(xm),inside=xm>XA&&xm<XB,dn=Math.min(Math.abs(xm-XA),Math.abs(xm-XB));
      // drawn in toward the nearer tie where it is pinched
      const xt=Math.abs(xm-XA)<Math.abs(xm-XB)?XA:XB,x=xm+(xt-xm)*.3*(1-g)*sstep(0,.03,h);
      const B0=at(ib,x);let y=B0.y-h*g;
      // a loose, uneven free edge (and the sides of the strip), a little turned
      const eB=sstep(.82,1,u),eS=sstep(.13,.18,Math.abs(xm));y+=.005*(nn4.n(xm*24,7)-.5)*eB;
      // the back face of the backrest, as measured off the chair: arched across, leaning back toward the top (smooth, so no slot or odd bit can catch the paper)
      let zr=Math.min(-.295+1.1*x*x-.26*(y-.80)-(y>.85?.6*(y-.85)**1.5:0),ENV(x,y))-.0035;
      const pin=Math.exp(-((dn/.012)**2));   // right under the tie: pressed flat to the plastic
      const pouch=inside?.028*pl(KX,(xm-XA)/(XB-XA)):.013*sstep(0,.08,dn);
      const swell=pouch*pl(KU,u)*(1-pin);
      const A=(.0065+.012*(1-g))*sstep(0,.01,h)*(1-pin);
      // crushed into small sharp creases right beside the tie
      const cr=sstep(.032,.008,dn)*(1-pin*.7)*.0022*(Math.pow(1-Math.abs(2*nn4.n(xm*38,u*16)-1),2)+.5*Math.pow(1-Math.abs(2*nn4.n(xm*60+9,u*28)-1),2)-.6);
      const edge=(eB*.006+eS*.004)*(nn4.n(xm*11+u*3,u*9+2)-.4);
      out.set(x,y,zr-swell-A*foldAt(xm,u)-cr-edge);
      // the top of the hang meets the sheet over the top edge
      const b=sstep(0,.03,h);out.lerpVectors(B0,out,b);return out};
    const CQ=new THREE.Vector3();
    const P=new Float32Array(NRu*NCu*3),UV=new Float32Array(NRu*NCu*2),N0=new THREE.Vector3(),Ta=new THREE.Vector3(),Tb=new THREE.Vector3(),Q=new THREE.Vector3();
    for(let i=0;i<NRu;i++)for(let j=0;j<NCu;j++){const s=i*dS,xm=(j/(NCu-1)-.5)*RIBW;
      // gathered in toward each tie
      let xs=xm,press=0,tent=0,fold=0,lift=0;
      for(const T of ties){const ws=sstep(T.sr-.035,T.sr,s)*(1-sstep(T.sf,T.sf+.035,s)),dx=xm-T.x0,adx=Math.abs(dx);
        xs+=-dx*.22*ws*Math.exp(-((dx/.035)**2));
        press=Math.max(press,ws*Math.exp(-((dx/.0062)**4)));tent=Math.max(tent,ws*Math.exp(-((dx/.017)**2)));
        // folds standing up out of the gathered paper, fanning out from the tie and dying away
        const A=.0065*ws*sstep(.004,.013,adx)*Math.exp(-adx/.045),ph=(s-T.st)/(.009+.32*adx)+T.seed+(dx>0?.37:0);fold+=A*(tri(ph)-.3);
}
      // between the ties the paper is drawn taut: a few straight tension ridges from tie to tie
      { const a=ties[0],b=ties[1];if(xm>a.x0+.01&&xm<b.x0-.01){const ws=sstep(a.sr,a.sr+.02,s)*(1-sstep(a.sf-.02,a.sf,s)),k=Math.sin(Math.PI*(xm-a.x0)/(b.x0-a.x0));fold+=ws*k*.0016*(tri((s-a.st)/.016+.2)-.5)}}
      const fade=1-sstep(SEND-.03,SEND-.004,s);
      Q.copy(at(i+lift/dS,xs));
      // the surface normal (outward) of the base sheet here
      const i1=Math.min(NRu-1,i+1),i0=Math.max(0,i-1),jj=Math.min(NCu-2,Math.max(0,j));Ta.subVectors(G0[i1][jj],G0[i0][jj]);Tb.subVectors(G0[i][Math.min(NCu-1,jj+1)],G0[i][Math.max(0,jj-1)]);N0.crossVectors(Tb,Ta).normalize();
      const d0=sdfAt(Q.x,Q.y,Q.z,GR),d=Math.max(0,d0-.0013);if(N0.dot(GR)<0)N0.negate();
      // the sheet lies on the plastic (a couple of millimetres of air under its folds), but still bridges the slots and the gaps
      const rest=Math.min(Math.max(0,d0-.0026),.011)*(1-sstep(.013,.02,d0))*fade,dd=Math.max(0,d-rest);
      let off=-rest+(-dd*Math.max(press,.72*tent)+fold*(1-press))*fade;off=Math.max(off,-d);
      Q.addScaledVector(N0,off);
      if(s<SB+.004){candy(xm,Math.min(s,SB),CQ);const w=sstep(SB+.004,SB-.012,s);Q.lerp(CQ,w);
        // never into the plastic
        const dq=sdfAt(Q.x,Q.y,Q.z,GR);if(window.__cdbg)window.__cdbg.push([i,j,+xm.toFixed(3),+s.toFixed(3),+CQ.x.toFixed(3),+CQ.y.toFixed(3),+CQ.z.toFixed(3),+dq.toFixed(4),+Q.z.toFixed(3)]);if(dq<.0022&&GR.lengthSq()>1e-12){GR.normalize();Q.addScaledVector(GR,.0022-dq)}}
      const o=(i*NCu+j)*3;P[o]=Q.x;P[o+1]=Q.y;P[o+2]=Q.z;UV[(i*NCu+j)*2]=j/(NCu-1);UV[(i*NCu+j)*2+1]=s/.42}
    // the last row meets the moving sheet edge to edge: there it follows the sheet's own straight edges exactly (no crack)
    for(let j=0;j<NCu;j++){const f=j/U,j0=Math.min(NC-2,Math.floor(f)),t=f-j0,a=B[NR-1][j0],b=B[NR-1][j0+1],o=((NRu-1)*NCu+j)*3;P[o]=a.x+(b.x-a.x)*t;P[o+1]=a.y+(b.y-a.y)*t;P[o+2]=a.z+(b.z-a.z)*t}
    const I=[];for(let i=0;i<NRu-1;i++)for(let j=0;j<NCu-1;j++){const a=i*NCu+j,b=a+1,c=a+NCu,d=c+1;I.push(a,c,b,b,c,d)}
    const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));g.setAttribute('uv',new THREE.BufferAttribute(UV,2));
    const UV1=new Float32Array(NRu*NCu*2);for(let i=0;i<NRu;i++)for(let j=0;j<NCu;j++){UV1[(i*NCu+j)*2]=j/(NCu-1);UV1[(i*NCu+j)*2+1]=i/(NRu-1)}g.setAttribute('uv1',new THREE.BufferAttribute(UV1,2));
    g.setIndex(I);g.computeVertexNormals();
    // drawn with crisp faces where the paper is folded and gathered behind (paper folds have sharp edges), smooth over the top and front
    const g2=g.toNonIndexed();{const pa=g2.attributes.position,na=g2.attributes.normal,u1=g2.attributes.uv1,a=new THREE.Vector3(),b=new THREE.Vector3(),c=new THREE.Vector3(),fn=new THREE.Vector3(),sn=new THREE.Vector3();
      for(let t=0;t<pa.count;t+=3){const sv=Math.max(u1.getY(t),u1.getY(t+1),u1.getY(t+2))*(NRu-1)*dS,w=sstep(SB+.012,SB-.01,sv);if(w<=0)continue;
        a.fromBufferAttribute(pa,t);b.fromBufferAttribute(pa,t+1);c.fromBufferAttribute(pa,t+2);fn.crossVectors(b.sub(a),c.sub(a)).normalize();
        for(let q=0;q<3;q++){sn.fromBufferAttribute(na,t+q);if(fn.dot(sn)<0)fn.negate();sn.lerp(fn,w).normalize();na.setXYZ(t+q,sn.x,sn.y,sn.z)}}}
    // a small ragged hole wherever a tie goes through the sheet
    const hc=canvas(256,Math.round(256*SEND/RIBW)),hx2=hc.getContext('2d'),hr=mulberry32(77);hx2.fillStyle='#fff';hx2.fillRect(0,0,hc.width,hc.height);hx2.fillStyle='#000';
    for(const T of ties)for(const sh of [T.sBack,T.sf-.004]){const cx=(T.x0/RIBW+.5)*hc.width+(hr()-.5)*2,cy=(1-sh/SEND)*hc.height,R=1.9+hr()*1.1,rot=hr()*3;hx2.beginPath();
      for(let k=0;k<=24;k++){const a=k/24*Math.PI*2,rr=R*(.6+.7*hr())*(1+.3*Math.sin(a*3+rot));hx2.lineTo(cx+Math.cos(a)*rr*1.3,cy+Math.sin(a)*rr)}hx2.fill();
      hx2.lineWidth=.9;hx2.strokeStyle='#000';for(let k=0;k<1+(hr()<.5?1:0);k++){let a=hr()*6.28,px=cx,py=cy;hx2.beginPath();hx2.moveTo(px,py);for(let q=0;q<3;q++){a+=(hr()-.5)*.9;px+=Math.cos(a)*1.6;py+=Math.sin(a)*1.6;hx2.lineTo(px,py)}hx2.stroke()}}
    const ht=new THREE.CanvasTexture(hc);ht.channel=1;ht.wrapS=ht.wrapT=THREE.ClampToEdgeWrapping;
    const m=ribM.clone();m.alphaMap=ht;m.alphaTest=.5;m.polygonOffset=false;m.needsUpdate=true;
    const mesh=new THREE.Mesh(g2,m);mesh.castShadow=true;mesh.receiveShadow=true;chairG.add(mesh);CAP.mesh=mesh;window.__CAP=CAP;
    // the ties themselves: a black strap pulled tight over the paper, into the hole at the back, through the slot, out of the hole in front
    const Pt=(i,j)=>{const o=(i*NCu+j)*3;return new THREE.Vector3(P[o],P[o+1],P[o+2])},Nt=(i,j)=>{const nn=g.attributes.normal,k=i*NCu+j;return new THREE.Vector3(nn.getX(k),nn.getY(k),nn.getZ(k))};
    for(const T of ties){const j0=Math.round((T.x0/RIBW+.5)*(NCu-1)),i0=Math.round(T.sBack/dS),i1=Math.round(T.sf/dS),pts=[];
      for(let i=i0;i<=i1;i+=2){const p=Pt(i,j0),n=Nt(i,j0);sdfAt(p.x,p.y,p.z,GR);if(n.dot(GR)<0)n.negate();
        // behind, the tie lies on top of the gathered bundle: on the outermost of the paper either side of it
        if(i*dS<SB){let zo=p.z;for(let jj=Math.max(0,j0-5);jj<=Math.min(NCu-1,j0+5);jj++)zo=Math.min(zo,Pt(i,jj).z);p.set(T.x0,p.y,zo-.0022);pts.push(p)}
        else pts.push(p.addScaledVector(n,.0022))}
      const A=pts[0],Bp=pts[pts.length-1],zb=A.z,zf=Bp.z,ym=T.ys-.006;T.dbg={A:A.toArray().map(v=>+v.toFixed(4)),B:Bp.toArray().map(v=>+v.toFixed(4)),ys:T.ys,x0:T.x0,sBack:+T.sBack.toFixed(3),sf:+T.sf.toFixed(3)};
      const hidden=[A,new THREE.Vector3(T.x0,ym,zb+.006),new THREE.Vector3(T.x0,ym-.002,(zb+zf)/2),new THREE.Vector3(T.x0,ym,zf-.006),Bp];
      const geo=mergeGeometries([strap(pts,.0027,.0008),strap(hidden,.0027,.0008)]);const tm=new THREE.Mesh(geo,tieM);tm.castShadow=true;chairG.add(tm);
      // the locking head on the front, a little above the hole, and the cut end sticking out of it
      const k=Math.max(1,pts.length-3),p1=pts[k],tg=V3.subVectors(pts[Math.min(pts.length-1,k+1)],pts[k-1]).normalize(),ang=Math.atan2(tg.y,tg.z);
      const head=new THREE.Mesh(new THREE.BoxGeometry(.0068,.005,.008),tieM);head.position.copy(p1);head.rotation.set(-ang,0,0);head.translateY(.0022);head.castShadow=true;chairG.add(head);
      const tail=new THREE.Mesh(new THREE.BoxGeometry(.0036,.0009,.012),tieM);tail.position.copy(p1);tail.rotation.set(-ang+1.05,.15*Math.sign(T.x0),0);tail.translateY(.004);tail.translateZ(.005);tail.castShadow=true;chairG.add(tail)}}
  // Catmull-Rom midpoint: (-a+9b+9c-d)/16
  function drawStrip(){const n=SR.n,nr=2*n-1;
    for(let r=0;r<n;r++){const ra=Math.max(0,r-1)*NCOL,rb=r*NCOL,rc=Math.min(n-1,r+1)*NCOL,rd=Math.min(n-1,r+2)*NCOL,o0=2*r*NCOL,o1=(2*r+1)*NCOL;
      for(let c=0;c<NCOL;c++)for(let k=0;k<3;k++){const b=PPOS[(rb+c)*3+k];TMP[(o0+c)*3+k]=b;if(r<n-1)TMP[(o1+c)*3+k]=(-PPOS[(ra+c)*3+k]+9*b+9*PPOS[(rc+c)*3+k]-PPOS[(rd+c)*3+k])*.0625}}
    for(let R=0;R<nr;R++){const arc=R*DS/2,v=arc/.42,v1=1-Math.min(1,arc/HOLEL),base=R*NCOL*3;
      for(let j=0;j<SUBC;j++){const c=j>>1,o=(R*SUBC+j)*3;
        if(!(j&1)){rPos[o]=TMP[base+c*3];rPos[o+1]=TMP[base+c*3+1];rPos[o+2]=TMP[base+c*3+2]}
        else{const ca=base+Math.max(0,c-1)*3,cb=base+c*3,cc=base+(c+1)*3,cd=base+Math.min(NCOL-1,c+2)*3;
          for(let k=0;k<3;k++)rPos[o+k]=(-TMP[ca+k]+9*TMP[cb+k]+9*TMP[cc+k]-TMP[cd+k])*.0625}
        const u=(R*SUBC+j)*2;rUv[u]=j/(SUBC-1);rUv[u+1]=v;rUv1[u]=j/(SUBC-1);rUv1[u+1]=v1}}
    // what a strip that lived on a roll and went through hands looks like: soft waves, wavy edges curling up, a few sharp creases
    for(let R=0;R<nr;R++){const arc=R*DS/2,Ra=Math.max(0,R-1),Rb=Math.min(nr-1,R+1);
      for(let j=0;j<SUBC;j++){const s=j/(SUBC-1)*2-1,ja=Math.max(0,j-1),jb=Math.min(SUBC-1,j+1);
        if(R<=2*(TIER+NPIN-1)+1)continue;let off=CRK(arc,s)*Math.min(1,(R-2*(TIER+NPIN-1)-1)/6);if(off===0)continue;
        const pa=(Ra*SUBC+j)*3,pb=(Rb*SUBC+j)*3,qa=(R*SUBC+ja)*3,qb=(R*SUBC+jb)*3;
        const ax=rPos[pb]-rPos[pa],ay=rPos[pb+1]-rPos[pa+1],az=rPos[pb+2]-rPos[pa+2],bx=rPos[qb]-rPos[qa],by=rPos[qb+1]-rPos[qa+1],bz=rPos[qb+2]-rPos[qa+2];
        let nx=ay*bz-az*by,ny=az*bx-ax*bz,nz=ax*by-ay*bx;const l=Math.hypot(nx,ny,nz);if(l<1e-12)continue;off/=l;
        const o=(R*SUBC+j)*3;OFS[o]=nx*off;OFS[o+1]=ny*off;OFS[o+2]=nz*off}}
    for(let i=0,m=nr*SUBC*3;i<m;i++){rPos[i]+=OFS[i];OFS[i]=0}
    for(let R=1;R<nr-1;R++)for(let j=0;j<SUBC;j++){const o=(R*SUBC+j)*3,a=o-SUBC*3,b=o+SUBC*3,mx=(rPos[a]+rPos[b])/2,my=(rPos[a+1]+rPos[b+1])/2,mz=(rPos[a+2]+rPos[b+2])/2;
      if(!Number.isFinite(rPos[o]+rPos[o+1]+rPos[o+2])||Math.hypot(rPos[o]-mx,rPos[o+1]-my,rPos[o+2]-mz)>.06){rPos[o]=mx;rPos[o+1]=my;rPos[o+2]=mz}}
    // the drawn surface between the simulated points can cut a corner of the chair: push any such point back out over the plastic
    { chairMat(CM);CMi.copy(CM).invert();const me=CMi.elements,mw=CM.elements;for(let i=0,m=nr*SUBC;i<m;i++){const o=i*3,x=rPos[o],y=rPos[o+1],z=rPos[o+2];
        const lx=me[0]*x+me[4]*y+me[8]*z+me[12],ly=me[1]*x+me[5]*y+me[9]*z+me[13],lz=me[2]*x+me[6]*y+me[10]*z+me[14];
        if(lx<-.43||lx>.43||ly>.96||ly<.02||lz<-.45||lz>.41)continue;const d=sdfAt(lx,ly,lz,GR),need=PT*.7;if(d>=need)continue;const gl=GR.length();if(gl<1e-6)continue;
        const f=(need-d)/gl;rPos[o]+=(mw[0]*GR.x+mw[4]*GR.y+mw[8]*GR.z)*f;rPos[o+1]+=(mw[1]*GR.x+mw[5]*GR.y+mw[9]*GR.z)*f;rPos[o+2]+=(mw[2]*GR.x+mw[6]*GR.y+mw[10]*GR.z)*f}}
    if(!CAP.built)buildCap();const r0=CAP.R0;ribGeo.setDrawRange(r0*(SUBC-1)*6,Math.max(0,nr-1-r0)*(SUBC-1)*6);ribGeo.attributes.position.needsUpdate=true;ribGeo.attributes.uv.needsUpdate=true;ribGeo.attributes.uv1.needsUpdate=true;
    ribGeo.computeVertexNormals();ribGeo.computeBoundingSphere()}

  /* ---------- the stone and the note crumpled around it ---------- */
  // the stone: a scanned rock. Its own colour stays; a fine red-granite grain (feldspar, glassy quartz, black mica) is laid over it in object space
  function graniteGrain(){const W=1024,c=canvas(W,W),x=c.getContext('2d'),r=mulberry32(71);x.fillStyle='#000';x.fillRect(0,0,W,W);x.globalCompositeOperation='lighter';
    const crystal=(rad,col,a)=>{const cx=r()*W,cy=r()*W,n=4+Math.floor(r()*4),pts=[];for(let k=0;k<n;k++){const an=k/n*6.283+r()*.7,rr=rad*(.5+r()*.7);pts.push([Math.cos(an)*rr,Math.sin(an)*rr*(.6+r()*.6)])}
      x.fillStyle=`rgba(${col},${a})`;for(const ox of [-W,0,W])for(const oy of [-W,0,W]){x.beginPath();pts.forEach((p,k)=>k?x.lineTo(cx+ox+p[0],cy+oy+p[1]):x.moveTo(cx+ox+p[0],cy+oy+p[1]));x.fill()}};
    for(let i=0;i<900;i++)crystal(14+r()*34,'255,0,0',.35+r()*.5);
    for(let i=0;i<1300;i++)crystal(6+r()*16,'0,255,0',.4+r()*.5);
    for(let i=0;i<2600;i++)crystal(1.5+r()*5,'0,0,255',.55+r()*.45);
    return c}
  const grT=tex(graniteGrain());
  const stoneMat=new THREE.MeshStandardMaterial({roughness:1,metalness:0,color:'#ffffff'});
  if(TX.stD){stoneMat.map=TX.stD;stoneMat.normalMap=TX.stN;stoneMat.roughnessMap=TX.stR;stoneMat.normalScale=new THREE.Vector2(1.1,1.1)}else stoneMat.color.set('#9a5446');
  stoneMat.onBeforeCompile=sh=>{sh.uniforms.tGr={value:grT};
    sh.vertexShader=sh.vertexShader.replace('#include <common>','#include <common>\nvarying vec3 vOP,vON;').replace('#include <begin_vertex>','#include <begin_vertex>\nvOP=position;vON=normal;');
    sh.fragmentShader=sh.fragmentShader.replace('#include <common>','#include <common>\nvarying vec3 vOP,vON;uniform sampler2D tGr;').replace('#include <map_fragment>',`#include <map_fragment>
      vec3 bw=pow(abs(normalize(vON)),vec3(4.0));bw/=bw.x+bw.y+bw.z;vec3 GP=vOP*5.5;
      vec3 gg=texture2D(tGr,GP.yz).rgb*bw.x+texture2D(tGr,GP.xz+.37).rgb*bw.y+texture2D(tGr,GP.xy+.71).rgb*bw.z;
      { vec3 c=diffuseColor.rgb;float lum=dot(c,vec3(.3,.59,.11));
        vec3 gf=texture2D(tGr,GP.yz*2.3+.13).rgb*bw.x+texture2D(tGr,GP.xz*2.3+.51).rgb*bw.y+texture2D(tGr,GP.xy*2.3+.29).rgb*bw.z;
        c=mix(c,c*vec3(1.13,.9,.83),gg.r*.62);c=mix(c,vec3(lum*1.02)*vec3(.96,.98,1.0),gg.g*.7);c*=1.0-.88*gg.b;c*=1.0-.5*gf.b;c=mix(c,vec3(lum*1.05),gf.g*.3);
        // keep the stone's own overall colour: the grain only redistributes it
        float l2=dot(c,vec3(.3,.59,.11));c*=mix(1.0,lum/max(l2,1e-3),.55);
        diffuseColor.rgb=c;}`).replace('#include <roughnessmap_fragment>','#include <roughnessmap_fragment>\nroughnessFactor*=1.0-.55*gg.b-.35*gg.g;')};
  stoneMat.customProgramCacheKey=()=>'granite3';
  let stoneGeo;
  if(TX.stoneObj){let g=null;TX.stoneObj.traverse(o=>{if(o.isMesh&&!g)g=o.geometry});g=g.clone();g.deleteAttribute('normal');g=mergeVertices(g,1e-5);
    g.computeBoundingBox();const c=g.boundingBox.getCenter(new THREE.Vector3());g.translate(-c.x,-c.y,-c.z);g.scale(.85,.85,.85);g.computeVertexNormals();stoneGeo=g}
  else{stoneGeo=new THREE.IcosahedronGeometry(1,5);stoneGeo.scale(SAX.x,SAX.y,SAX.z)}
  const stone=new THREE.Mesh(stoneGeo,stoneMat);stone.castShadow=true;stone.receiveShadow=true;scene.add(stone);

  try{await Promise.race([document.fonts.load('700 196px Arimo'),new Promise(r=>setTimeout(r,1500))])}catch(e){}
  const noteC=canvas(1024,732),nx=noteC.getContext('2d');nx.drawImage(tPaper,0,0,1024,732);
  nx.fillStyle='rgba(255,255,255,.25)';nx.fillRect(0,0,1024,732);
  nx.fillStyle='#141412';nx.font='700 196px "Helvetica Neue", Helvetica, Arimo, Arial, sans-serif';nx.textBaseline='alphabetic';
  if('letterSpacing' in nx)nx.letterSpacing='-6px';
  nx.fillText('ЕЩЁ',58,236);nx.fillText('НЕ',58,436);nx.fillText('КОНЕЦ',58,636);   // printed the right way round on the face that shows it
  const NW=.42,NH=.30,noteMat=new THREE.MeshStandardMaterial({map:tex(noteC,{srgb:true}),roughness:.9,side:THREE.DoubleSide});
  noteMat.onBeforeCompile=sh=>{sh.uniforms.uFace={value:NB.uFace||1};noteMat.userData.sh=sh;sh.fragmentShader=sh.fragmentShader.replace('#include <common>','#include <common>\nuniform float uFace;').replace('#include <map_fragment>',
    `vec4 sampledDiffuseColor=(gl_FrontFacing==(uFace>0.))?texture2D(map,vMapUv):vec4(vec3(.93,.925,.9)*(.97+.03*texture2D(map,vMapUv*.13+.81).g),1.);diffuseColor*=sampledDiffuseColor;`)};
  noteMat.customProgramCacheKey=()=>'note1face';
  const note=new THREE.Mesh(new THREE.PlaneGeometry(NW,NH,84,60),noteMat);
  note.castShadow=true;note.receiveShadow=true;note.frustumCulled=false;note.matrixAutoUpdate=false;scene.add(note);
  // what is drawn: the same sheet as separate triangles, each with its own normal worked out here (crisp paper facets, and a
  // degenerate sliver can never produce a broken normal on any graphics card)
  scene.remove(note);const noteR=new THREE.Mesh(note.geometry.toNonIndexed(),noteMat);noteR.castShadow=true;noteR.receiveShadow=true;noteR.frustumCulled=false;noteR.matrixAutoUpdate=false;noteR.userData.ptFace=()=>NB.uFace||1;scene.add(noteR);
  const nIdx=note.geometry.index.array,rP=noteR.geometry.attributes.position,rN=noteR.geometry.attributes.normal;
  function noteDraw(){const A=nP.array,B=rP.array,Nn=rN.array;let lx=0,ly=1,lz=0;
    for(let t=0;t<nIdx.length;t+=3){const a=nIdx[t]*3,b=nIdx[t+1]*3,c=nIdx[t+2]*3,o=t*3;
      B[o]=A[a];B[o+1]=A[a+1];B[o+2]=A[a+2];B[o+3]=A[b];B[o+4]=A[b+1];B[o+5]=A[b+2];B[o+6]=A[c];B[o+7]=A[c+1];B[o+8]=A[c+2];
      const ux=A[b]-A[a],uy=A[b+1]-A[a+1],uz=A[b+2]-A[a+2],vx=A[c]-A[a],vy=A[c+1]-A[a+1],vz=A[c+2]-A[a+2];let nx=uy*vz-uz*vy,ny=uz*vx-ux*vz,nz=ux*vy-uy*vx;const l=Math.hypot(nx,ny,nz);
      if(l>1e-12&&Number.isFinite(l)){nx/=l;ny/=l;nz/=l;lx=nx;ly=ny;lz=nz}else{nx=lx;ny=ly;nz=lz}
      for(let q=0;q<9;q+=3){Nn[o+q]=nx;Nn[o+q+1]=ny;Nn[o+q+2]=nz}}
    rP.needsUpdate=true;rN.needsUpdate=true}
  const nP=note.geometry.attributes.position,nU=note.geometry.attributes.uv,nN=nP.count,nn=makeNoise(57),nn2=makeNoise(61),noteInfo=[];
  // crumpled paper: sharp ridges from a few folding directions, kept as a memory when it is smoothed out again
  const ridge=(u,v,f,o)=>{const a=nn.n(u*f+o,v*f-o);return Math.pow(1-Math.abs(2*a-1),3)};
  // the stone's own cross-section, measured from its mesh: radius from its long axis, by angle and by position along it
  const RT_NA=64,RT_NX=28,RT_X0=-.068,RT_DX=.136/(RT_NX-1),RTAB=new Float32Array(RT_NA*RT_NX);
  { const p=stoneGeo.attributes.position;for(let i=0;i<p.count;i++){const x=p.getX(i),y=p.getY(i),z=p.getZ(i),a=((Math.atan2(z,y)/(Math.PI*2))%1+1)%1,ia=Math.round(a*RT_NA)%RT_NA,ix=Math.round((x-RT_X0)/RT_DX);
      if(ix<0||ix>=RT_NX)continue;const k=ix*RT_NA+ia;RTAB[k]=Math.max(RTAB[k],Math.hypot(y,z))}
    for(let pass=0;pass<6;pass++)for(let ix=0;ix<RT_NX;ix++)for(let ia=0;ia<RT_NA;ia++){const k=ix*RT_NA+ia;if(RTAB[k]>0)continue;let s=0,n=0;
      for(const [dx,da] of [[0,1],[0,-1],[1,0],[-1,0]]){const jx=ix+dx,ja=(ia+da+RT_NA)%RT_NA;if(jx<0||jx>=RT_NX)continue;const v=RTAB[jx*RT_NA+ja];if(v>0){s+=v;n++}}if(n)RTAB[k]=s/n*(pass<3?1:.97)}
    const tmp=RTAB.slice();for(let ix=0;ix<RT_NX;ix++)for(let ia=0;ia<RT_NA;ia++){let s=0;for(let d=-1;d<=1;d++)s+=tmp[ix*RT_NA+(ia+d+RT_NA)%RT_NA];RTAB[ix*RT_NA+ia]=Math.max(tmp[ix*RT_NA+ia],s/3)}}
  const rAt=(a,x)=>{const fa=((a/(Math.PI*2))%1+1)%1*RT_NA,fx=Math.min(RT_NX-1.001,Math.max(0,(x-RT_X0)/RT_DX)),ia=Math.floor(fa),ix=Math.floor(fx),ta=fa-ia,tx=fx-ix,ib=(ia+1)%RT_NA;
    return (RTAB[ix*RT_NA+ia]*(1-ta)+RTAB[ix*RT_NA+ib]*ta)*(1-tx)+(RTAB[(ix+1)*RT_NA+ia]*(1-ta)+RTAB[(ix+1)*RT_NA+ib]*ta)*tx};
  // the note goes round the stone about twice along its long axis (blank end inside, the words outside), the spare paper at both ends is twisted shut
  let PC=0;for(let k=0;k<32;k++)PC+=rAt(k/32*Math.PI*2,0)+.0032;PC=PC/32*Math.PI*2;
  // how far round the stone each bit of the note sits: walked along the stone's own outline, one paper thickness further out on every turn
  const WS=[0],WA=[.6];{let s=0,a=.6;while(s<NW+.01){const R=rAt(a,0)+.0032+(a-.6)/(Math.PI*2)*.0032;s+=R*.01;a+=.01;WS.push(s);WA.push(a)}}
  const angOfS=s=>{let lo=0,hi=WS.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(WS[m]<=s)lo=m;else hi=m}const t=Math.min(1,Math.max(0,(s-WS[lo])/(WS[hi]-WS[lo]||1)));return WA[lo]+(WA[hi]-WA[lo])*t};
  const sOfAng=a=>{let lo=0,hi=WA.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(WA[m]<=a)lo=m;else hi=m}const t=Math.min(1,Math.max(0,(a-WA[lo])/(WA[hi]-WA[lo]||1)));return WS[lo]+(WS[hi]-WS[lo])*t};
  const ANG=L=>angOfS(NW-L),LAYMAX=(angOfS(NW)-.6)/(Math.PI*2);
  for(let i=0;i<nN;i++)noteInfo.push({x:nP.getX(i),y:nP.getY(i),u:nU.getX(i),v:nU.getY(i),wrap:new THREE.Vector3(),mem:0});
  // wrapped about twice round the stone's long axis, following its real outline; the spare paper at both ends gathered and twisted shut.
  // A fresh crumple every throw: the ridge pattern is offset by a new random seed
  // a crumpled sheet is a set of flat facets joined along sharp creases: random crease lines, a new set every throw
  const NCR=[];
  /* a crumpled-and-smoothed sheet is a set of flat facets with straight sharp folds between them, some points where several folds
     meet in a star. Built as a random triangulation of the sheet with a height at every corner: the height field is flat on each
     facet and folds sharply along every edge. The edges are also the creases the simulated sheet bends along. New every throw. */
  const NTRI=[];
  function delaunay(P){const st=[[-10,-10],[10,-10],[0,10]],pts=P.concat(st),n=P.length;let T=[[n,n+1,n+2]];
    const cc=t=>{const [a,b,c]=t.map(i=>pts[i]),d=2*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]));const A=a[0]*a[0]+a[1]*a[1],B=b[0]*b[0]+b[1]*b[1],C=c[0]*c[0]+c[1]*c[1];
      const ux=(A*(b[1]-c[1])+B*(c[1]-a[1])+C*(a[1]-b[1]))/d,uy=(A*(c[0]-b[0])+B*(a[0]-c[0])+C*(b[0]-a[0]))/d;return [ux,uy,(a[0]-ux)**2+(a[1]-uy)**2]};
    let C=T.map(cc);
    for(let i=0;i<n;i++){const p=pts[i],bad=[],keep=[],keepC=[];T.forEach((t,k)=>{const c=C[k];if((p[0]-c[0])**2+(p[1]-c[1])**2<c[2])bad.push(t);else{keep.push(t);keepC.push(c)}});
      const E=new Map();for(const t of bad)for(const [a,b] of [[t[0],t[1]],[t[1],t[2]],[t[2],t[0]]]){const k=a<b?a+','+b:b+','+a;E.set(k,E.has(k)?null:[a,b])}
      T=keep;C=keepC;for(const e of E.values())if(e){const t=[e[0],e[1],i];T.push(t);C.push(cc(t))}}
    return T.filter(t=>t[0]<n&&t[1]<n&&t[2]<n)}
  function noteCreases(seed){const r=mulberry32(seed*13+5);NCR.length=0;NTRI.length=0;
    const P=[],H=[];
    // corners and edge points (the edges of a smoothed sheet stay a little lifted), then the inside: a few star centres with points round them
    for(const [x,y] of [[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])P.push([x*NW,y*NH]);
    for(let k=0;k<7;k++){P.push([(r()-.5)*NW,-NH/2],[(r()-.5)*NW,NH/2])}for(let k=0;k<4;k++){P.push([-NW/2,(r()-.5)*NH],[NW/2,(r()-.5)*NH])}
    const NB=P.length;
    for(let k=0;k<4;k++){const cx=(r()-.5)*NW*.8,cy=(r()-.5)*NH*.75;P.push([cx,cy]);const m=4+Math.floor(r()*3);for(let q=0;q<m;q++){const a=q/m*6.283+r()*.6,L=.035+r()*.05;P.push([cx+Math.cos(a)*L,cy+Math.sin(a)*L*.85])}}
    for(let k=0;k<16;k++)P.push([(r()-.5)*NW*.94,(r()-.5)*NH*.92]);
    for(const p of P){p[0]=Math.max(-NW/2,Math.min(NW/2,p[0]));p[1]=Math.max(-NH/2,Math.min(NH/2,p[1]))}
    P.forEach((p,i)=>H.push(i<NB?.006+r()*.016:(r()<.3?.016+r()*.016:r()*.012)));
    for(const t of delaunay(P)){const [a,b,c]=t.map(i=>P[i]),[ha,hb,hc]=t.map(i=>H[i]);const d=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(d)<1e-9)continue;NTRI.push({a,b,c,ha,hb,hc,d})}
    // every inner edge is a crease
    const seen=new Set();for(const T of NTRI)for(const [p,q] of [[T.a,T.b],[T.b,T.c],[T.c,T.a]]){const k=[p,q].map(v=>v[0].toFixed(5)+':'+v[1].toFixed(5)).sort().join('|');if(seen.has(k))continue;seen.add(k);
      if((Math.abs(p[0])>NW/2-1e-4&&Math.abs(q[0])>NW/2-1e-4)||(Math.abs(p[1])>NH/2-1e-4&&Math.abs(q[1])>NH/2-1e-4))continue;
      const px=(p[0]+q[0])/2,py=(p[1]+q[1])/2,L=Math.hypot(q[0]-p[0],q[1]-p[1])/2,a=Math.atan2(q[1]-p[1],q[0]-p[0]);
      NCR.push({px,py,nx:-Math.sin(a),ny:Math.cos(a),tx:Math.cos(a),ty:Math.sin(a),len:L,amp:0,w:.01,ang:.35+r()*.9})}}
  // the height of the crumpled sheet at a point of it: flat on each facet
  function facetH(x,y){for(const T of NTRI){const l1=((T.b[1]-T.c[1])*(x-T.c[0])+(T.c[0]-T.b[0])*(y-T.c[1]))/T.d,l2=((T.c[1]-T.a[1])*(x-T.c[0])+(T.a[0]-T.c[0])*(y-T.c[1]))/T.d,l3=1-l1-l2;
      if(l1>=-1e-6&&l2>=-1e-6&&l3>=-1e-6)return l1*T.ha+l2*T.hb+l3*T.hc}return 0}
  const crossesCrease=(x1,y1,x2,y2)=>{for(const c of NCR){const d1=(x1-c.px)*c.nx+(y1-c.py)*c.ny,d2=(x2-c.px)*c.nx+(y2-c.py)*c.ny;if(d1*d2<0){const mx=(x1+x2)/2-c.px,my=(y1+y2)/2-c.py;if(Math.abs(mx*c.tx+my*c.ty)<c.len)return c}}return null};
  // paper round a stone does not follow every dip: it spans flat from one high point to the next, with a sharp fold at each
  const FAC=[];
  function wrapFacets(r){FAC.length=0;let a=-12-r()*.4;while(a<32){FAC.push([a,.0006+r()*.0032]);a+=.55+r()*.6}}
  const facetR=(ang,x,lay)=>{let k=0;while(k<FAC.length-2&&FAC[k+1][0]<=ang)k++;const [a1,b1]=FAC[k],[a2,b2]=FAC[k+1];
    const r1=rAt(a1,x)+.0032+lay*.0032+b1,r2=rAt(a2,x)+.0032+lay*.0032+b2;
    // the straight line between the two corner points, met by the ray at this angle
    const y1=r1*Math.cos(a1),z1=r1*Math.sin(a1),y2=r2*Math.cos(a2),z2=r2*Math.sin(a2),cy=Math.cos(ang),cz=Math.sin(ang),ey=y2-y1,ez=z2-z1,den=cy*ez-cz*ey;
    return Math.abs(den)<1e-9?r1:(y1*ez-z1*ey)/den};
  const tri=x=>Math.abs((x-Math.floor(x))-.5)*2;
  function buildNote(seed,a0=0){const r=mulberry32(seed),o1=r()*50,o2=r()*50,o3=r()*50,o4=r()*50,tw=.6+r()*.8;wrapFacets(mulberry32(seed*3+1));
    const S2=[0],A2=[.6+a0];{let q=0,a=.6+a0;while(q<NW+.01){const R=rAt(a,0)+.0032+(a-.6-a0)/(Math.PI*2)*.0032;q+=R*.01;a+=.01;S2.push(q);A2.push(a)}}
    const angW=q=>{let lo=0,hi=S2.length-1;while(hi-lo>1){const m=(lo+hi)>>1;if(S2[m]<=q)lo=m;else hi=m}const t=Math.min(1,Math.max(0,(q-S2[lo])/(S2[hi]-S2[lo]||1)));return A2[lo]+(A2[hi]-A2[lo])*t};
    for(let i=0;i<nN;i++){const ni=noteInfo[i],u=ni.u,v=ni.v;
      const s=(1-u)*NW,xa=(v-.5)*NH,lim=.05,ax=Math.abs(xa),e=Math.max(0,(ax-lim)/(NH/2-lim));
      const xe=Math.sign(xa)*(ax<lim?ax:lim+.03*e-.012*e*e);
      const ang0=angW(s),ang=ang0+e*e*tw*Math.sign(xa)*.6,lay=(ang0-.6-a0)/(Math.PI*2);
      const c1=ridge(u,v,5,o1),c2=ridge(u,v,9,o2),c3=ridge(u,v,17,o3),c4=ridge(u,v,31,o4);
      // crisp folds, not soft wrinkles: the crease net pressed in, a little fine crinkle, sharp V-pleats where the ends are twisted
      const fh0=facetH(ni.x,ni.y);
      const crum=.0003*(c3+c4)+.3*fh0;
      const pleat=e*(.003+.007*tri(ang*1.4+nn2.n(u*3+o1,v*3)*1.5));
      const Rf=Math.max(facetR(ang,xe,Math.max(0,lay)),rAt(ang,xe)+.0032+Math.max(0,lay)*.0032);
      const R=Rf*(1-.72*e*e)+crum+pleat;
      ni.wrap.set(xe,R*Math.cos(ang),R*Math.sin(ang));
      // what stays in the paper when it opens out: sharp ridges and valleys of the crumple, both ways
      // sharp ridges and valleys along the creases, flat in between; a little fine crinkle
      ni.mem=fh0-.008+.0004*(c4-.3)}}
  noteCreases(7);buildNote(7);

  // ---------- story timeline ----------
  const camShot=new THREE.Vector3(-.15,.62,.75);
  let unwrapT0=-1,rollT=-1,restHold=0,noteO=new THREE.Vector3(),noteUp=new THREE.Vector3(),noteRight=new THREE.Vector3();
  const STRIP0={};   // the strip as it rests before anything happens
  function resetBodies(){for(const f of FEET){f.jn=0;f.jt.set(0,0,0)}CH.p.copy(CH_REST.p);CH.q.copy(CH_REST.q);CH.v.set(0,0,0);CH.w.set(0,0,0);updIw(CH);CMp.copy(chairMat(CM));
    Object.assign(RS,{ph:0,zr:EDGE.z-.017,vz:0,ang:0,om:0,r:R0,U:0,s:0,landed:false,bump:0,hE:0,tension:0,lastD:0});RS.head.copy(fdir);
    ST.active=false;STB.p.set(0,-5,0);STB.v.set(0,0,0);STB.w.set(0,0,0)}
  resetBodies();layStrip();SR.restAtSpawn=SR.lastRest;
  for(let k=0;k<300;k++){stripStep(1/240)}buildTies();   // let the strip settle onto the chair before anyone sees it
  STRIP0.P=PPOS.slice(0,SR.n*NCOL*3);STRIP0.W=PIM.slice(0,SR.n*NCOL);STRIP0.n=SR.n;STRIP0.rest=SR.lastRest;STRIP0.U0=SR.U0;
  // aim the throw once, with exactly the physics the story runs: the launch speed that brings it to the front-right leg slowly
  { CH.fixed=false;updIw(CH);let best=null;const res={};
    // a hard throw through the fortochka: it lands once on the planks a little short of the chair and strikes the front leg on the bounce.
    // The exact throw is found here, on this device, with the story's own physics, so what is chosen is what will be played
    const tryT=(vh,ax)=>{setAim(ax,.28);resetBodies();launch(vh);let touches=0,was=false,t1=-1,k=0;
      for(;k<4*480;k++){physStep(1/480,0);if(ST.contact&&!was){touches++;if(t1<0)t1=k}was=ST.contact;if(ST.hit)break}
      const ok=ST.hit&&ST.leg===0&&touches>=1&&touches<=3&&t1>=0&&(k-t1)/480<.7;if(!ok){SEARCH.push([+vh.toFixed(2),ax,-1]);return -1}
      // let it come to rest exactly as in the story, and see where: it should end up against the front of the leg, propping it
      for(let j=0;j<480*1.5;j++){physStep(1/480,0);if(j>60&&STB.v.length()<.02&&ST.contact)break}
      const L=toLocal(CH,STB.p,new THREE.Vector3()),d=Math.hypot(L.x-LEGS[0][0],L.z-(LEGS[0][1]+.062));SEARCH.push([+vh.toFixed(2),ax,0,+ST.hitV.toFixed(2),touches,+d.toFixed(3)]);
      return d+(ST.hitV>2.6?.05:0)};
    for(const ax of [.3,.29,.31,.28,.32])for(let vh=5.9;vh<=6.66;vh+=.05){const sc=tryT(vh,ax);if(sc>=0&&(!best||sc<best.sc))best={vh,ax,sc};if(best&&best.sc<.012)break}
    if(best)setAim(best.ax,.28);else setAim(.3,.28);
    THROW=best?best.vh:6.25;CH.fixed=false;resetBodies();stoneVel.set(aimDir.x*THROW,-1,aimDir.z*THROW).normalize()}

  /* ---------- the note opening: a stiff crumpled sheet springing loose and falling open ----------
     Paper is not cloth: it does not flow, it springs and flops. On the knock the crumpled wrap loosens at once (the outer turn first);
     once the stone lies still, the sheet falls open onto the boards, outer end first, each part swinging round the stone (never through
     it) to where it comes to lie. Where it lies: flat on the boards with the stone sitting on its inner face (so the words come up),
     still crumpled into flat facets and sharp folds, the innermost bit still curled up against the stone; placed clear of the chair's
     feet, of the paper strip and roll in front of the chair, and of the walls. */
  const NB={on:false,t0:0,tf:-1,F:null,c:new THREE.Vector3(),fly:null};
  const _a=new THREE.Vector3(),_b=new THREE.Vector3(),_d=new THREE.Vector3(),_ax=new THREE.Vector3(),_qb=new THREE.Quaternion();
  function bloomStart(t){NB.on=true;NB.t0=t;NB.tf=-1;NB.F=null;
    // the loose end of the wrap is wherever the paper slipped round on the way: on the upper side, toward the room
    const tc=V1.set(camShot.x-STB.p.x,0,camShot.z-STB.p.z).normalize().add(V3.set(0,0,1.6)).normalize().multiplyScalar(.55).add(V2.set(0,.85,0)).applyQuaternion(_qi.copy(STB.q).invert());
    buildNote(NS.seed,Math.atan2(tc.z,tc.y)-angOfS(NW)+.5);
    // which side of the sheet faces the stone: the words are printed on that (inner) side
    const k=nsV(14,10)*1,ni=noteInfo[k],nx=noteInfo[k+1],ny=noteInfo[k-85];   // rows of the drawn sheet run top to bottom
    _a.subVectors(nx.wrap,ni.wrap);_b.subVectors(ny.wrap,ni.wrap);_d.crossVectors(_a,_b);
    V2.set(0,ni.wrap.y,ni.wrap.z);NB.frontOut=_d.dot(V2)<0;
    NB.uFace=window.__UF||1;/* the print is on the side that faces the stone in the wrap */const sh=noteMat.userData.sh;if(sh)sh.uniforms.uFace.value=NB.uFace}
  // the place on the floor where the opened sheet will lie
  function bloomFinal(){const c=NB.c.copy(STB.p),fl0=fh(c.x,c.z);
    // keep the flat sheet's orientation continuous with the wrap's: the inner face ends up on top
    const mirror=window.__MR||-1;   // opened out, that side ends up on top   // front face down when it faced out
    const legs=LEGS.map(([lx,lz])=>toWorld(CH,V3.set(lx,0,lz),new THREE.Vector3()));
    const pen=(X,Z)=>{let p=0;toLocal(CH,V3.set(X,.02,Z),V4);
      if(Math.abs(V4.x)<.225&&V4.z>.2)p+=1;                            // the strip and the roll run forward from the seat
      for(const L of legs){const d=Math.hypot(X-L.x,Z-L.z);if(d<.04)p+=4}
      if(X<X0+.02||X>X1-.02||Z<Z0+.02||Z>Z1-.02)p+=2;return p};
    // where the paper was when wrapped (plan), to keep the motion short
    const W=[];for(let j=0;j<=60;j+=4)for(let i=0;i<=84;i+=4){const ni=noteInfo[j*85+i];V1.copy(ni.wrap).applyQuaternion(STB.q).add(STB.p);W.push([ni.x,ni.y,V1.x,V1.z])}
    let best=null;
    for(const sa of [.035,.05,.07,.09])for(const ya of [-.1,-.065,-.03,0,.03,.065,.1])for(let k=0;k<36;k++){const th=k/36*Math.PI*2,c1=Math.cos(th),s1=Math.sin(th);
      const ax=NW/2-sa,ay=ya;const map=(x,y)=>{const u=x-ax,v=(y-ay)*mirror;return [c.x+c1*u-s1*v,c.z+s1*u+c1*v]};
      let P=0,M=0;for(const [x,y,wx,wz] of W){const [X,Z]=map(x,y);P+=pen(X,Z);M+=Math.hypot(X-wx,Z-wz)}
      const sc=P*10+M/W.length+Math.abs(ya)*.3;if(!best||sc<best.sc)best={sc,c1,s1,ax,ay}}
    const {c1,s1,ax,ay}=best;NB.best=best;
    const F=NB.F=new Float32Array(nN*3);NB.lift=0;NB.y0=STB.p.y;
    for(let i=0;i<nN;i++){const ni=noteInfo[i],sv=NW/2-ni.x,u=ni.x-ax,v=(ni.y-ay)*mirror,X=c.x+c1*u-s1*v,Z=c.z+s1*u+c1*v;
      // still crumpled: flat facets and sharp folds standing up off the boards, pressed flat under and round the stone
      const dS=Math.hypot(X-c.x,Z-c.z),press=sstep(.04,.085,dS);
      let Y=fh(X,Z)+.0008+facetH(ni.x,ni.y)*2.1*press;
      // under the stone the paper is pressed flat; if the stone's own underside comes lower than the paper, the stone rests on it
      V2.set(X,Y,Z);for(let q=0;q<40&&V2.y>fh(X,Z)+.0008&&inStone(V2,.0025);q++)V2.y=Math.max(fh(X,Z)+.0008,V2.y-.001);Y=V2.y;
      { let up=0;while(up<.03&&inStone(V2.set(X,Y-up,Z),.0025))up+=.0005;NB.lift=Math.max(NB.lift,up)}
      F[i*3]=X;F[i*3+1]=Y;F[i*3+2]=Z;
      // the innermost bit stays curled up against the stone
      const keep=sstep(.035,.012,sv);if(keep>0){V1.copy(ni.wrap).applyQuaternion(STB.q).add(STB.p);F[i*3]+=(V1.x-X)*keep;F[i*3+1]+=(V1.y-Y)*keep;F[i*3+2]+=(V1.z-Z)*keep}}
    NB.fl=fl0}
  function inStone(P,m){V3.copy(P).sub(STB.p).applyQuaternion(_qi.copy(STB.q).invert());const ax=Math.abs(V3.x);if(ax>=.078)return false;
    let rs=rAt(Math.atan2(V3.z,V3.y),Math.max(-.066,Math.min(.066,V3.x)))+m;if(ax>.066){const q=(ax-.066)/.012;rs*=Math.sqrt(Math.max(0,1-q*q))}return Math.hypot(V3.y,V3.z)<rs}
  function stoneOut(P){V3.copy(P).sub(STB.p).applyQuaternion(_qi.copy(STB.q).invert());const ax=Math.abs(V3.x);
    if(ax<.078){const rr=Math.hypot(V3.y,V3.z);let rs=rAt(Math.atan2(V3.z,V3.y),Math.max(-.066,Math.min(.066,V3.x)))+.0018;if(ax>.066){const q=(ax-.066)/.012;rs*=Math.sqrt(Math.max(0,1-q*q))}
      if(rr<rs&&rr>1e-6){V3.y*=rs/rr;V3.z*=rs/rr;P.copy(V3.applyQuaternion(STB.q).add(STB.p))}}}
  // where a point of the sheet is meant to be at time t (the guide; the real sheet follows it as an unstretchable sheet, below)
  function bloomAt(i,t,out){const ni=noteInfo[i],sv=(NW/2-ni.x)/NW,tl=Math.max(0,t-NB.t0),
      lo=1-Math.exp(-tl*3.2)*(Math.cos(tl*9)+.35*Math.sin(tl*9)),ee=Math.max(0,(Math.abs(ni.v-.5)*NH-.05)/(NH/2-.05));
    out.copy(ni.wrap);const gr=1+lo*(.03+.16*sv*sv)*(1-.85*ee);out.y*=gr;out.z*=gr;out.applyQuaternion(STB.q).add(STB.p);
    if(NB.F&&NB.tf>=0){const k=i*3,d=.35+.9*(1-sv)+.1*Math.sin(ni.y*31+NS.seed),x=Math.min(1,Math.max(0,(t-NB.tf-d)/2.1));
      // a flap of stiff paper toppling over: slow to start, gathering speed under its own weight, landing with a little speed left
      let e=x*x*(2.2-1.2*x);
      if(e>0){_a.subVectors(out,STB.p);_b.set(NB.F[k]-STB.p.x,NB.F[k+1]-STB.p.y,NB.F[k+2]-STB.p.z);const ra=_a.length(),rb=_b.length();_a.normalize();_b.normalize();
        const om=Math.acos(Math.max(-1,Math.min(1,_a.dot(_b))));const r=ra+(rb-ra)*e+.045*Math.sin(Math.PI*Math.min(1,e*1.15))*sv;
        // turn round the stone about a fixed axis (a plain slerp blows up when start and end lie on opposite sides of the stone)
        _ax.crossVectors(_a,_b);if(_ax.lengthSq()<1e-6){_ax.crossVectors(_a,V4.set(0,1,0));if(_ax.lengthSq()<1e-6)_ax.crossVectors(_a,V4.set(1,0,0))}_ax.normalize();
        _d.copy(_a).applyQuaternion(_qb.setFromAxisAngle(_ax,om*e));
        if(Number.isFinite(_d.x)&&Number.isFinite(r))out.copy(STB.p).addScaledVector(_d,r)}}
    return out}
  function bloomPose(t){for(let i=0;i<nN;i++){bloomAt(i,t,V1);stoneOut(V1);const f=fh(V1.x,V1.z)+.0006;if(V1.y<f)V1.y=f;nP.setXYZ(i,V1.x,V1.y,V1.z)}}
  /* once the stone lies still the sheet is a real sheet: particles that cannot stretch, stiff between creases, hinged along them,
     colliding with the stone, the boards, the chair and itself, led gently along the guide so it ends where it should */
  function bloomSim(t){const P=NS.P,O=NS.O;
    for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i,v=nsV(i,j);P[k*3]=O[k*3]=nP.getX(v);P[k*3+1]=O[k*3+1]=nP.getY(v);P[k*3+2]=O[k*3+2]=nP.getZ(v)}
    const C=NS.C=[],RX=k=>noteInfo[nsV(k%NSX,(k/NSX)|0)].x,RY=k=>noteInfo[nsV(k%NSX,(k/NSX)|0)].y;
    const add=(a,b,L,k)=>C.push([a,b,L,L,k,-1]);
    for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i;
      if(i<NSX-1)add(k,k+1,NDX,1);if(j<NSY-1)add(k,k+NSX,NDY,1);
      if(i<NSX-1&&j<NSY-1){const dg=Math.hypot(NDX,NDY);add(k,k+NSX+1,dg,1);add(k+1,k+NSX,dg,1)}
      // bending: a facet stays a stiff flat plate; across a crease it hinges freely
      for(const [di,dj] of [[2,0],[0,2],[2,2],[2,-2]]){const i2=i+di,j2=j+dj;if(i2<0||i2>=NSX||j2<0||j2>=NSY)continue;const m=j2*NSX+i2,L=Math.hypot(di*NDX,dj*NDY),c=crossesCrease(RX(k),RY(k),RX(m),RY(m));
        add(k,m,c?L*Math.cos(c.ang/2):L,c?.04:.5)}}
    NS.qi.copy(STB.q).invert();NS.pp.copy(STB.p);NS.pq.copy(STB.q);NS.t=0;NS.calm=0;NS.frozen=false;NS.tk=-1;NS.T=null;NS.on=true;NB.simT=t;
    // the final shape's own sharp facets, kept as detail on top of the smooth sheet
    const crF=new Float32Array(NSN*3);for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i,v=nsV(i,j);crF[k*3]=NB.F[v*3];crF[k*3+1]=NB.F[v*3+1];crF[k*3+2]=NB.F[v*3+2]}
    const save=P.slice();P.set(crF);noteSurface();NB.res=new Float32Array(nN*3);for(let i=0;i<nN;i++){NB.res[i*3]=NB.F[i*3]-nP.getX(i);NB.res[i*3+1]=NB.F[i*3+1]-nP.getY(i);NB.res[i*3+2]=NB.F[i*3+2]-nP.getZ(i)}P.set(save)}
  function storyInit(){NS.on=false;NB.on=false;NB.F=null;NS.seed=STILL?11+(+(new URLSearchParams(location.search).get('seed'))||0):(Math.random()*1e6)|0;noteCreases(NS.seed);buildNote(NS.seed);resetBodies();PPOS.set(STRIP0.P);POLD.set(STRIP0.P);PIM.fill(0);PIM.set(STRIP0.W);SR.n=STRIP0.n;SR.lastRest=STRIP0.rest;SR.restAtSpawn=STRIP0.rest;SR.U0=STRIP0.U0;SR.frozenTo=0;ROWF.fill(0);
    unwrapT0=-1;rollT=-1;restHold=0;launch(THROW);ST.active=false;STB.p.set(0,-5,0)}
  let acc=0;
  function physStep(h,t){
    if(window.__rec&&ST.active){const R=window.__rec;R.k=(R.k||0)+1;if(R.k%48===0)R.out.push([R.k,+STB.p.x.toFixed(4),+STB.p.y.toFixed(4),+STB.p.z.toFixed(4),+CH.p.x.toFixed(4),ST.contact?1:0])}
    if(ST.active&&!ST.kin)stoneForces(h);if(ST.hit&&ST.damp>0&&!ST.contact){STB.v.multiplyScalar(Math.exp(-h*10));STB.w.multiplyScalar(Math.exp(-h*10))}
    // a stone lying on the planks cannot keep spinning on the spot: paper and grit on the boards grip it (the solver has only one contact point, so it would)
    if(ST.active&&!ST.kin&&ST.contact){const k=Math.exp(-h*(ST.hit?9:3));STB.w.y*=k;if(ST.hit){STB.w.multiplyScalar(Math.exp(-h*2.5));STB.v.multiplyScalar(Math.exp(-h*1.5))}}
    CH.v.y-=G*h;if(RS.ph===2)RLB.v.y-=G*h;
    rollStep(h);
    nCT=0;chairContacts();
    if(ST.active&&ST.kin&&(!ST.settle||t>ST.settle.t0+ST.settle.T)){V4.set(1,0,0).applyQuaternion(STB.q);
      for(const kk of [-.035,0,.035]){V3.copy(STB.p).addScaledVector(V4,kk);toLocal(CH,V3,V1);const d=sdfAt(V1.x,V1.y,V1.z,GR),rs=.036;
        if(d<rs&&GR.lengthSq()>1e-8){const nW=_t3.copy(GR).normalize().applyQuaternion(CH.q).clone();const pt=V3.clone().addScaledVector(nW,-d);addContact(CH,null,pt,nW.negate(),Math.min(.002,rs-d),.4,0,5)}}}
    if(ST.active&&!ST.kin)stoneContacts(()=>{if(S.breakPending){S.breakPending=false;breakPane(transom.pane,STB.p);S.shake=Math.max(S.shake,.45);S.gust=Math.max(S.gust,1.1);window.dispatchEvent(new CustomEvent('glass'))}});
    rollAirContacts();
    for(const b of [CH,STB,RLB])updIw(b);
    solveContacts(h,10);
    // the paper-wrapped stone against a flexible plastic leg: the blow is soaked up, it barely rebounds
    if(ST.absorb){ST.absorb=0;STB.v.multiplyScalar(.16);STB.w.multiplyScalar(.3)}
    integrate(CH,h);if(ST.active&&!ST.kin)integrate(STB,h);if(RS.ph===2)integrate(RLB,h);
    CH.w.multiplyScalar(Math.exp(-h*.4));
  }
  const _q2=new THREE.Quaternion();
  const SPTS=(()=>{const p=stoneGeo.attributes.position,o=[];for(let i=0;i<p.count;i+=2)o.push(p.getX(i),p.getY(i),p.getZ(i));return new Float32Array(o)})();
  // lowest point of the stone itself in a given attitude (relative to its centre)
  function stoneMinY(q){const m=V2.set(0,1,0).applyQuaternion(_qi.copy(q).invert());let lo=1;for(let i=0;i<SPTS.length;i+=3){const y=SPTS[i]*m.x+SPTS[i+1]*m.y+SPTS[i+2]*m.z;if(y<lo)lo=y}return lo}

  /* ---------- the note as a real sheet once the stone has stopped ----------
     A 29 x 21 grid of particles (every third vertex of the drawn sheet). Paper does not stretch; it bends, and it remembers how it was bent:
     rest lengths start as the wrapped shape and relax toward the opened, crumpled sheet (curl memory tighter on the inner turn), so the
     wrap springs loose by itself, the free end falls away, the stone sits on what is left. It collides with the stone, the planks,
     the chair and itself; friction holds it where it lies. Each throw gets its own random crumple. */
  const NSX=29,NSY=21,NSN=NSX*NSY,NDX=NW/(NSX-1),NDY=NH/(NSY-1);
  const NS={on:false,P:new Float32Array(NSN*3),O:new Float32Array(NSN*3),C:[],t:0,calm:0,frozen:false,seed:1,qi:new THREE.Quaternion(),pp:new THREE.Vector3(),pq:new THREE.Quaternion(),pqi:new THREE.Quaternion(),y0:0,yT:0,tk:-1};
  const nsV=(i,j)=>(3*j)*85+3*i;
  function noteInit(){const P=NS.P,O=NS.O,r=mulberry32(NS.seed*7+3);
    // the loose end of the wrap is wherever the paper slipped round on the way: put it on the upper side, toward the room
    { const tc=V1.set(camShot.x-STB.p.x,0,camShot.z-STB.p.z).normalize().add(V3.set(0,0,1.6)).normalize().multiplyScalar(.55).add(V2.set(0,.85,0)).applyQuaternion(_qi.copy(STB.q).invert());
      buildNote(NS.seed,Math.atan2(tc.z,tc.y)-angOfS(NW)+.5)}
    // every bit of the paper is still travelling with the stone when it starts to open
    const hN=3/240;for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i,ni=noteInfo[nsV(i,j)];V1.copy(ni.wrap).applyQuaternion(STB.q);V4.crossVectors(STB.w,V1).add(STB.v).multiplyScalar(hN);V1.add(STB.p);
      P[k*3]=V1.x;P[k*3+1]=V1.y;P[k*3+2]=V1.z;O[k*3]=V1.x-V4.x;O[k*3+1]=V1.y-V4.y;O[k*3+2]=V1.z-V4.z}
    const C=NS.C=[],d=(a,b)=>Math.hypot(P[a*3]-P[b*3],P[a*3+1]-P[b*3+1],P[a*3+2]-P[b*3+2]);
    const add=(a,b,fin,k,del)=>C.push([a,b,d(a,b),fin,k,del]);
    const RX=k=>noteInfo[nsV(k%NSX,(k/NSX)|0)].x,RY=k=>noteInfo[nsV(k%NSX,(k/NSX)|0)].y;
    // paper does not shrink: a span keeps its length unless a crease runs across it, where the fold takes some of it up
    const span=(a,b,L)=>{const c=crossesCrease(RX(a),RY(a),RX(b),RY(b));return c?L*Math.cos(c.ang*.35):L*(1-.002*r())};
    for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i,u=i/(NSX-1),v=1-j/(NSY-1),e=Math.max(0,(Math.abs(v-.5)*NH-.05)/(NH/2-.05));
      // the outer end lets go first, the inner turn last; the twisted ends a little later still
      const del=e>0?.0+.08*r():.02+.3*u+.08*r();
      if(i<NSX-1)add(k,k+1,span(k,k+1,NDX),1,del);if(j<NSY-1)add(k,k+NSX,span(k,k+NSX,NDY),1,del);
      if(i<NSX-1&&j<NSY-1){const dg=Math.hypot(NDX,NDY);add(k,k+NSX+1,span(k,k+NSX+1,dg),1,del);add(k+1,k+NSX,span(k+1,k+NSX,dg),1,del)}
      // bending: stiff across a facet (paper stays flat, with only the curl it was given round the stone), soft where a crease runs, which stays folded
      if(i<NSX-2){const c=crossesCrease(RX(k),RY(k),RX(k+2),RY(k+2)),L=2*NDX;
        if(c)add(k,k+2,L*Math.cos(c.ang/2),.05,del);else{const Rm=.06+2.4*Math.pow(1-u,1.6)+.1*r();add(k,k+2,2*Rm*Math.sin(L/(2*Rm)),1,del)}}
      if(j<NSY-2){const c=crossesCrease(RX(k),RY(k),RX(k+2*NSX),RY(k+2*NSX)),L=2*NDY;add(k,k+2*NSX,c?L*Math.cos(c.ang/2):L,c?.05:1,del)}
      if(i<NSX-2&&j<NSY-2){const c=crossesCrease(RX(k),RY(k),RX(k+2*NSX+2),RY(k+2*NSX+2)),L=2*Math.hypot(NDX,NDY);add(k,k+2*NSX+2,c?L*Math.cos(c.ang/2):L,c?.05:.6,del)}
      // paper is stiff: between two creases a facet stays a flat plate (with only its curl), so long braces span every facet; the creases are the hinges
      if(e===0){const Rm=.06+2.4*Math.pow(1-u,1.6),cx=n=>2*Rm*Math.sin(n*NDX/(2*Rm));
        for(const [di,dj] of [[3,0],[0,3],[3,3],[3,-3],[4,1],[1,4],[4,-1],[-1,4]]){const i2=i+di,j2=j+dj;if(i2<0||i2>=NSX||j2<0||j2>=NSY)continue;const m=j2*NSX+i2;
          if(crossesCrease(RX(k),RY(k),RX(m),RY(m)))continue;add(k,m,Math.hypot(cx(Math.abs(di)),Math.abs(dj)*NDY),.9,del)}}}
    // the crumpled wrap was only held by the throw; now it springs: the outer turn outward from the stone's axis, harder the farther out
    { const ax=V2.set(1,0,0).applyQuaternion(STB.q);for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=(j*NSX+i)*3,u=i/(NSX-1);
        V1.set(P[k]-STB.p.x,P[k+1]-STB.p.y,P[k+2]-STB.p.z);V1.addScaledVector(ax,-V1.dot(ax));const l=V1.length()||1;const sp=(.85+.45*r())*Math.pow(1-u,.9)*(1/120);
        O[k]-=V1.x/l*sp;O[k+1]-=V1.y/l*sp-(1-u)*.15/120;O[k+2]-=V1.z/l*sp}}
    NS.qi.copy(STB.q).invert();NS.pp.copy(STB.p);NS.pq.copy(STB.q);NS.t=0;NS.calm=0;NS.frozen=false;NS.on=true;NS.tk=-1;NS.T=null}
  // once the stone lies still it settles down onto the paper the wrap has let go of under it
  function noteSettle(){NS.y0=STB.p.y;NS.yT=Math.min(NS.y0,fh(STB.p.x,STB.p.z)-stoneMinY(STB.q)+.0052);NS.tk=NS.t;noteLayout()}
  /* where the opened sheet comes to lie: flat on the boards, the way it already mostly lies (best fit of the sheet's own plan to where
     its paper is now, turned or turned over), under the stone, clear of the chair leg. The sheet is then drawn there gently, as paper
     that has sprung open slowly flattens out; it still collides with the stone, the floor and itself the whole time. */
  function noteLayout(){const P=NS.P,M=[];let mx=0,my=0,px=0,pz=0;
    for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const ni=noteInfo[nsV(i,j)],k=j*NSX+i;M.push([ni.x,ni.y]);mx+=ni.x;my+=ni.y;px+=P[k*3];pz+=P[k*3+2]}
    mx/=NSN;my/=NSN;px/=NSN;pz/=NSN;
    let best=null;for(const fl of (window.__FL?[window.__FL]:[1,-1])){let a=0,b=0;for(let k=0;k<NSN;k++){const u=M[k][0]-mx,v=(M[k][1]-my)*fl,X=P[k*3]-px,Z=P[k*3+2]-pz;a+=u*X+v*Z;b+=u*Z-v*X}
      const th=Math.atan2(b,a),c=Math.cos(th),sn=Math.sin(th);let e=0;for(let k=0;k<NSN;k++){const u=M[k][0]-mx,v=(M[k][1]-my)*fl,X=c*u-sn*v,Z=sn*u+c*v;e+=(X-(P[k*3]-px))**2+(Z-(P[k*3+2]-pz))**2}
      if(fl<0)e/=2.5;   // the printed side up, unless that would mean turning the whole sheet over
      if(!best||e<best.e)best={e,c,sn,fl}}
    const {c,sn,fl}=best;let tx=px,tz=pz;
    const toW=(u,v)=>[tx+c*(u-mx)-sn*(v-my)*fl,tz+sn*(u-mx)+c*(v-my)*fl],toM=(X,Z)=>{const dx=X-tx,dz=Z-tz;return [mx+c*dx+sn*dz,my+(-sn*dx+c*dz)*fl]};
    // the stone must lie on the sheet (a little in from its edge), and the chair's foot must not stand on it: the edge runs between them
    { const L=toWorld(CH,V3.set(LEGS[0][0],0,LEGS[0][1]),new THREE.Vector3());
      for(let it=0;it<40;it++){let moved=false;
        { const [u,v]=toM(STB.p.x,STB.p.z),cu=Math.max(-NW/2+.03,Math.min(NW/2-.03,u)),cv=Math.max(-NH/2+.03,Math.min(NH/2-.03,v));
          if(cu!==u||cv!==v){const [X0,Z0]=toW(u,v),[X1,Z1]=toW(cu,cv);tx+=(X0-X1)*.5;tz+=(Z0-Z1)*.5;moved=true}}
        { const [u,v]=toM(L.x,L.z);if(Math.abs(u)<NW/2+.02&&Math.abs(v)<NH/2+.02){const dx=STB.p.x-L.x,dz=STB.p.z-L.z,d=Math.hypot(dx,dz)||1;tx+=dx/d*.004;tz+=dz/d*.004;moved=true}}
        if(!moved)break}}
    NS.T=new Float32Array(NSN*3);for(let k=0;k<NSN;k++){const [X,Z]=toW(M[k][0],M[k][1]);NS.T[k*3]=X;NS.T[k*3+2]=Z;NS.T[k*3+1]=fh(X,Z)+.0009+.9*facetH(M[k][0],M[k][1])}NS.tT=NS.t}   // it keeps the facets and folds it was crumpled into
  const nsHash=new Map();
  window.__nan=()=>{let n=0,f=-1;const A=nP.array;for(let i=0;i<A.length;i++)if(!Number.isFinite(A[i])){n++;if(f<0)f=i}let np=0;for(let k=0;k<NS.P.length;k++)if(!Number.isFinite(NS.P[k]))np++;let nf=0;if(NB.F)for(let k=0;k<NB.F.length;k++)if(!Number.isFinite(NB.F[k]))nf++;let nr=0;if(NB.res)for(let k=0;k<NB.res.length;k++)if(!Number.isFinite(NB.res[k]))nr++;return [n,f,np,nf,nr]};
  window.__nbDbg=()=>{let x=0,y=0,z=0;for(let k=0;k<NSN;k++){x+=NS.P[k*3];y+=NS.P[k*3+1];z+=NS.P[k*3+2]}return {on:NS.on,nb:NB.on,tf:NB.tf,t:+NS.t.toFixed(2),c:[x/NSN,y/NSN,z/NSN].map(v=>+v.toFixed(3)),stone:STB.p.toArray().map(v=>+v.toFixed(3)),kin:!!ST.kin}};
  window.__nsStretch=()=>{if(!NS.on)return null;const P=NS.P;let mx=0,at=null;for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i;for(const [m,L] of [[i<NSX-1?k+1:-1,NDX],[j<NSY-1?k+NSX:-1,NDY]]){if(m<0)continue;const d=Math.hypot(P[k*3]-P[m*3],P[k*3+1]-P[m*3+1],P[k*3+2]-P[m*3+2])/L;if(d>mx){mx=d;const cc=NS.C.find(c=>c[0]===k&&c[1]===m);at=[i,j,m-k,+P[k*3].toFixed(3),+P[k*3+1].toFixed(3),+P[k*3+2].toFixed(3),cc&&[+cc[2].toFixed(4),+cc[3].toFixed(4),cc[4]]]}}}return [+mx.toFixed(2),at,NS.frozen,+NS.t.toFixed(2)]};
  const _rg=new THREE.Vector3();
  function noteStep(h){if(!NS.on||NS.frozen)return;NS.t+=h;const P=NS.P,O=NS.O,t=NS.t;
    if(NS.tk>=0)STB.p.y=NS.y0+(NS.yT-NS.y0)*sstep(0,1.1,t-NS.tk);
    NS.qi.copy(STB.q).invert();NS.pqi.copy(NS.pq).invert();
    const dmp=Math.exp(-h*2.6);let mv=0;
    for(let k=0;k<NSN;k++){const a=k*3;for(let c=0;c<3;c++){const vv=(P[a+c]-O[a+c])*dmp;O[a+c]=P[a+c];P[a+c]+=vv+(c===1?-G*h*h:0)}}
    // self: layers of the sheet never pass through each other
    if(!(NS.T&&t-NS.tT>.7)){nsHash.clear();const cs=.006,key=(x,y,z)=>((Math.floor(x/cs)*73856093)^(Math.floor(y/cs)*19349663)^(Math.floor(z/cs)*83492791));
    for(let k=0;k<NSN;k++){const kk=key(P[k*3],P[k*3+1],P[k*3+2]);let l=nsHash.get(kk);if(!l)nsHash.set(kk,l=[]);l.push(k)}
    for(let k=0;k<NSN;k++){const x=P[k*3],y=P[k*3+1],z=P[k*3+2],ik=k%NSX,jk=(k/NSX)|0;
      for(let dx=-1;dx<=1;dx++)for(let dy=-1;dy<=1;dy++)for(let dz=-1;dz<=1;dz++){const l=nsHash.get(key(x+dx*cs,y+dy*cs,z+dz*cs));if(!l)continue;
        for(const m of l){if(m<=k)continue;const im=m%NSX,jm=(m/NSX)|0;if(Math.abs(im-ik)<=2&&Math.abs(jm-jk)<=2)continue;
          const ex=P[m*3]-P[k*3],ey=P[m*3+1]-P[k*3+1],ez=P[m*3+2]-P[k*3+2],dd=Math.hypot(ex,ey,ez),mn=.0026;
          if(dd<mn&&dd>1e-7){const f=(mn-dd)/dd*.5;P[k*3]-=ex*f;P[k*3+1]-=ey*f;P[k*3+2]-=ez*f;P[m*3]+=ex*f;P[m*3+1]+=ey*f;P[m*3+2]+=ez*f}}}}
    }
    const C=NS.C;
    for(let it=0;it<8;it++){
      for(const c of C){const a=c[0]*3,b=c[1]*3,rel=sstep(0,1,(t-c[5])/.5),rest=c[2]+(c[3]-c[2])*rel;
        const dx=P[b]-P[a],dy=P[b+1]-P[a+1],dz=P[b+2]-P[a+2],dd=Math.hypot(dx,dy,dz);if(dd<1e-8)continue;
        const f=(dd-rest)/dd*.5*c[4];P[a]+=dx*f;P[a+1]+=dy*f;P[a+2]+=dz*f;P[b]-=dx*f;P[b+1]-=dy*f;P[b+2]-=dz*f}
      // the stone (its own outline), the planks (with friction), the walls
      for(let k=0;k<NSN;k++){const a=k*3;V1.set(P[a]-STB.p.x,P[a+1]-STB.p.y,P[a+2]-STB.p.z).applyQuaternion(NS.qi);const ax=Math.abs(V1.x);
        if(ax<.078){const rr=Math.hypot(V1.y,V1.z),an=Math.atan2(V1.z,V1.y);let rs=rAt(an,Math.max(-.066,Math.min(.066,V1.x)))+.0013;if(ax>.066){const q=(ax-.066)/.012;rs*=Math.sqrt(Math.max(0,1-q*q))}
          if(rr<rs+.0005&&rr>1e-6){if(rr<rs){V1.y*=rs/rr;V1.z*=rs/rr;V1.applyQuaternion(STB.q).add(STB.p);P[a]=V1.x;P[a+1]=V1.y;P[a+2]=V1.z}
            // paper on rough stone grips: where it touches, it is carried with the stone and does not slide round it
            _rg.set(O[a]-NS.pp.x,O[a+1]-NS.pp.y,O[a+2]-NS.pp.z).applyQuaternion(NS.pqi).applyQuaternion(STB.q).add(STB.p);
            const mu=P[a+1]>STB.p.y?.012:.05;P[a]+=(_rg.x-O[a]-(P[a]-O[a]))*mu;P[a+1]+=(_rg.y-O[a+1]-(P[a+1]-O[a+1]))*mu;P[a+2]+=(_rg.z-O[a+2]-(P[a+2]-O[a+2]))*mu}}
        const fl=fh(P[a],P[a+2])+.0007;if(P[a+1]<fl){P[a+1]=fl;P[a]-=(P[a]-O[a])*.14;P[a+2]-=(P[a+2]-O[a+2])*.14}
        if(P[a]<X0+.003)P[a]=X0+.003;if(P[a]>X1-.003)P[a]=X1-.003;if(P[a+2]<Z0+.003)P[a+2]=Z0+.003;if(P[a+2]>Z1-.003)P[a+2]=Z1-.003}}
    // the chair legs
    for(let k=0;k<NSN;k++){const a=k*3;if(P[a+1]>.5)continue;V2.set(P[a],P[a+1],P[a+2]);toLocal(CH,V2,V3);const d=sdfAt(V3.x,V3.y,V3.z,GR);
      if(d<.0016&&GR.lengthSq()>1e-10){V4.copy(GR).normalize().applyQuaternion(CH.q);P[a]+=V4.x*(.0016-d);P[a+1]+=V4.y*(.0016-d);P[a+2]+=V4.z*(.0016-d)}}
    if(NB.on){const tt=NB.simT+t,kp=.025+.09*sstep(1.6,3.8,t);for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=(j*NSX+i)*3;bloomAt(nsV(i,j),tt,V2);P[k]+=(V2.x-P[k])*kp;P[k+1]+=(V2.y-P[k+1])*kp;P[k+2]+=(V2.z-P[k+2])*kp}}
    for(let k=0;k<NSN*3;k++)if(!Number.isFinite(P[k])){P[k]=O[k]=Number.isFinite(O[k])?O[k]:STB.p.getComponent(k%3)}
    if(NB.on)for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=(j*NSX+i)*3;if(Math.hypot(P[k]-STB.p.x,P[k+1]-STB.p.y,P[k+2]-STB.p.z)>.5){bloomAt(nsV(i,j),NB.simT+t,V2);P[k]=O[k]=V2.x;P[k+1]=O[k+1]=V2.y;P[k+2]=O[k+2]=V2.z}}
    // paper cannot stretch: no span between neighbouring points may end up longer than it is (whatever a collision did)
    for(let it=0;it<3;it++)for(const c of C){if(c[4]<.9)continue;const a=c[0]*3,b=c[1]*3,rl=sstep(0,1,(t-c[5])/.5),mx=(c[2]+(c[3]-c[2])*rl)*1.04;
      const dx=P[b]-P[a],dy=P[b+1]-P[a+1],dz=P[b+2]-P[a+2],dd=Math.hypot(dx,dy,dz);if(dd<=mx)continue;const f=(dd-mx)/dd*.5;P[a]+=dx*f;P[a+1]+=dy*f;P[a+2]+=dz*f;P[b]-=dx*f;P[b+1]-=dy*f;P[b+2]-=dz*f}
    for(let j=0;j<NSY;j++)for(let i=0;i<NSX;i++){const k=j*NSX+i,a=k*3;let sx=0,sy=0,sz=0,n=0,far=0;
      for(const [di,dj] of [[1,0],[-1,0],[0,1],[0,-1]]){const i2=i+di,j2=j+dj;if(i2<0||i2>=NSX||j2<0||j2>=NSY)continue;const m=(j2*NSX+i2)*3;sx+=P[m];sy+=P[m+1];sz+=P[m+2];n++;
        if(Math.hypot(P[m]-P[a],P[m+1]-P[a+1],P[m+2]-P[a+2])>2.2*NDX)far++}
      if(n&&far===n){P[a]=O[a]=sx/n;P[a+1]=O[a+1]=sy/n;P[a+2]=O[a+2]=sz/n}}
    for(let k=0;k<NSN*3;k++){const dv=Math.abs(P[k]-O[k]);if(dv>mv)mv=dv}
    NS.pp.copy(STB.p);NS.pq.copy(STB.q);
    NS.calm=mv<h*.004?NS.calm+h:0;if(t>(NB.on?5:2.5)&&(NS.tk>=0||NB.on)&&NS.calm>.6)NS.frozen=true}
  // smooth (Catmull-Rom) surface through the sheet's particles, for every drawn vertex
  const crw=(t)=>{const t2=t*t,t3=t2*t;return [-.5*t3+t2-.5*t,1.5*t3-2.5*t2+1,-1.5*t3+2*t2+.5*t,.5*t3-.5*t2]};
  const CRW=[0,1,2].map(f=>crw(f/3));
  function noteSurface(){const P=NS.P;
    for(let vy=0;vy<=60;vy++){const j0=Math.min(NSY-2,Math.floor(vy/3)),fj=vy-3*j0,wj=fj===3?[0,0,1,0]:CRW[fj];
      for(let vx=0;vx<=84;vx++){const i0=Math.min(NSX-2,Math.floor(vx/3)),fi=vx-3*i0,wi=fi===3?[0,0,1,0]:CRW[fi];let x=0,y=0,z=0;
        for(let b=0;b<4;b++){const jj=Math.max(0,Math.min(NSY-1,j0-1+b));if(!wj[b])continue;for(let a=0;a<4;a++){if(!wi[a])continue;const ii=Math.max(0,Math.min(NSX-1,i0-1+a)),w=wi[a]*wj[b],k=(jj*NSX+ii)*3;x+=P[k]*w;y+=P[k+1]*w;z+=P[k+2]*w}}
        nP.setXYZ(vy*85+vx,x,y,z)}}}
  function storyStep(dt){
    // fixed little steps, so live playback and the stills follow the same physics
    acc+=dt;const H=1/240;
    while(acc>=H-1e-9){acc-=H;const t=S.storyT-acc;
      physStep(H/2,t);physStep(H/2,t);
      stripStep(H);if(NS.on&&(NS.sub=(NS.sub||0)+1)%3===0)noteStep(3*H)}
    freezeRows();
    const t=S.storyT;
    // the stone has come to rest against the chair: the paper opens, then the roll goes
    const sv=STB.v.length()+STB.w.length()*.05;
    // the blow on the leg jars the crumpled wrap loose: it starts to open at once, while the stone is still rolling to a stop
    if(ST.active&&ST.hit&&unwrapT0<0&&t>ST.hitT+.03){unwrapT0=t;bloomStart(t)}
    if(NB.on&&!ST.kin){restHold=sv<.05&&ST.contact?restHold+dt:0;if(restHold>.1||t>ST.hitT+.7){ST.kin=true;STB.v.set(0,0,0);STB.w.set(0,0,0);NB.tf=t;bloomFinal();bloomPose(t);bloomSim(t)}}
    if(NB.on&&NB.F&&NB.tf>=0)STB.p.y=NB.y0+NB.lift*sstep(1.2,3,t-NB.tf);
    // the knock on the leg jolts the chair: the roll, sitting right on the lip, tips over at once
    if(ST.hit&&rollT<0)rollT=ST.hitT+.04;
    if(ST.active&&!ST.hit&&t>12&&rollT<0)rollT=t;
    if(rollT>=0&&t>=rollT&&RS.ph===0){RS.ph=1;RS.vz=.22}
  }
  // when nothing is happening the strip still lives a little in the draught
  function idleStrip(dt){acc+=dt;const H=1/240;let k=0;while(acc>=H-1e-9&&k<8){acc-=H;stripStep(H);k++}if(k>=8)acc=0}
  const NBAD=new Uint8Array(85*61);
  function storyPose(){
    const t=S.storyT;
    chairMat(CM);chairG.matrix.copy(CM);chairG.matrixWorldNeedsUpdate=true;
    side.set(1,0,0).transformDirection(CM);
    for(let i=0;i<4;i++){V1.set(LEGS[i][0],0,LEGS[i][1]).applyMatrix4(CM);OCC.uOcc.value[i].set(V1.x,V1.z,.04,.5)}
    V1.set(0,0,-.02).applyMatrix4(CM);OCC.uOcc.value[4].set(V1.x,V1.z,.3,.2);
    const R=RS;rollFrame();
    roll.scale.set(R.r,1,R.r);roll.position.copy(R.c);
    roll.quaternion.setFromUnitVectors(Yax,R.axis);_qi.setFromAxisAngle(Yax,-R.ang);roll.quaternion.multiply(_qi);
    if(R.ph>=3&&R.c.y<R.r+.05){const h=RIBW/2;OCC.uCapA.value[1].set(R.c.x-R.axis.x*h,R.c.z-R.axis.z*h,R.c.x+R.axis.x*h,R.c.z+R.axis.z*h);OCC.uCapB.value[1].set(R.r*.75,.5,0,0)}else OCC.uCapB.value[1].y=0;
    drawStrip();
    /* stone and the note */
    stone.visible=note.visible=noteR.visible=t>=0&&ST.active;
    OCC.uOcc.value[5].w=t>=0&&ST.active?.55*cl(1-(STB.p.y-.06)/.08):0;OCC.uOcc.value[5].x=STB.p.x;OCC.uOcc.value[5].y=STB.p.z;OCC.uOcc.value[5].z=.07;
    if(t<0||!ST.active)return;
    stone.visible=unwrapT0>=0;stone.position.copy(STB.p);stone.quaternion.copy(STB.q);
    note.matrix.identity();const wv=V1;
    if(NB.on&&NS.on){noteSurface();const e=sstep(.8,3.6,NS.t);for(let i=0;i<nN;i++){V1.set(nP.getX(i)+NB.res[i*3]*e,nP.getY(i)+NB.res[i*3+1]*e,nP.getZ(i)+NB.res[i*3+2]*e);stoneOut(V1);const f=fh(V1.x,V1.z)+.0006;if(V1.y<f)V1.y=f;nP.setXYZ(i,V1.x,V1.y,V1.z)}}
    else if(NB.on)bloomPose(t);
    else if(!NS.on){for(let i=0;i<nN;i++){wv.copy(noteInfo[i].wrap).applyQuaternion(STB.q).add(STB.p);nP.setXYZ(i,wv.x,wv.y,wv.z)}}
    else{noteSurface();note.geometry.computeVertexNormals();const nr=note.geometry.attributes.normal,km=sstep(.1,1.4,NS.t);
      // the fine creases of the crumple, on top of the simulated shape
      for(let i=0;i<nN;i++){const m=noteInfo[i].mem*km;let x=nP.getX(i)+nr.getX(i)*m,y=nP.getY(i)+nr.getY(i)*m,z=nP.getZ(i)+nr.getZ(i)*m;const f=fh(x,z)+.0006;if(y<f)y=f;
        V3.set(x-STB.p.x,y-STB.p.y,z-STB.p.z).applyQuaternion(NS.qi);const ax=Math.abs(V3.x);
        if(ax<.078){const rr=Math.hypot(V3.y,V3.z);let rs=rAt(Math.atan2(V3.z,V3.y),Math.max(-.066,Math.min(.066,V3.x)))+.0016;if(ax>.066){const q=(ax-.066)/.012;rs*=Math.sqrt(Math.max(0,1-q*q))}
          if(rr<rs&&rr>1e-6){V3.y*=rs/rr;V3.z*=rs/rr;V3.applyQuaternion(STB.q).add(STB.p);x=V3.x;y=V3.y;z=V3.z}}
        nP.setXYZ(i,x,y,z)}}
    // last line of defence: a point of the sheet can never stand far out from its neighbours or far from the stone (no spikes, on any device)
    { const A=nP.array,bad=NBAD;bad.fill(0);let nb=0;
      for(let j=0;j<=60;j++)for(let i=0;i<=84;i++){const v=j*85+i,o=v*3;if(!Number.isFinite(A[o]+A[o+1]+A[o+2])||Math.hypot(A[o]-STB.p.x,A[o+2]-STB.p.z)>.5){bad[v]=1;nb++;continue}
        for(const [di,dj] of [[1,0],[0,1]]){const ii=i+di,jj=j+dj;if(ii>84||jj>60)continue;const q=(jj*85+ii)*3;if(Math.hypot(A[o]-A[q],A[o+1]-A[q+1],A[o+2]-A[q+2])>.03){
          // the one of the two that is farther from the stone is the wrong one
          const da=Math.hypot(A[o]-STB.p.x,A[o+1]-STB.p.y,A[o+2]-STB.p.z),db=Math.hypot(A[q]-STB.p.x,A[q+1]-STB.p.y,A[q+2]-STB.p.z);const w=da>db?v:jj*85+ii;if(!bad[w]){bad[w]=1;nb++}}}}
      if(nb>(window.__nbadMax||0))window.__nbadMax=nb;
      for(let pass=0;pass<40&&nb>0;pass++){let fixed=0;for(let v=0;v<nN;v++){if(!bad[v])continue;const i=v%85,j=(v/85)|0;let sx=0,sy=0,sz=0,n=0;
          for(const [di,dj] of [[1,0],[-1,0],[0,1],[0,-1]]){const ii=i+di,jj=j+dj;if(ii<0||ii>84||jj<0||jj>60)continue;const w=jj*85+ii;if(bad[w])continue;sx+=A[w*3];sy+=A[w*3+1];sz+=A[w*3+2];n++}
          if(n){A[v*3]=sx/n;A[v*3+1]=sy/n;A[v*3+2]=sz/n;bad[v]=2;fixed++}}
        for(let v=0;v<nN;v++)if(bad[v]===2){bad[v]=0;nb--}if(!fixed)break}}
    nP.needsUpdate=true;noteDraw();
  }
  storyInit();
  window.__chairG=chairG;window.__ribbon=ribbon;window.__search=()=>SEARCH;window.__chairState=()=>{const up=new THREE.Vector3(0,1,0).applyQuaternion(CH.q);const fw=new THREE.Vector3(0,0,1).applyQuaternion(CH.q);return {t:+S.storyT.toFixed(2),tilt:+(Math.acos(Math.min(1,up.y))*57.3).toFixed(2),pitch:+(Math.asin(-fw.y)*57.3).toFixed(2),roll:+(Math.asin(new THREE.Vector3(1,0,0).applyQuaternion(CH.q).y)*57.3).toFixed(2),dp:CH.p.clone().sub(CH_REST.p).toArray().map(v=>+(v*1000).toFixed(1)),ph:RS.ph,ten:+RS.tension.toFixed(1),v:+CH.v.length().toFixed(3)}};window.__sdf=()=>({len:SDF.d?SDF.d.length:-1,W:SDF.W,s1:sdfAt(0,.46,0),s2:sdfAt(0,.6,0),s3:sdfAt(0,.444,0),s4:sdfAt(.3,.1,.275),tx:!!TX.sdf});
  window.__strip=()=>{const out=[];for(let r=0;r<SR.n;r+=2){const i=(r*NCOL+4)*3;const l=toLocal(CH,new THREE.Vector3(PPOS[i],PPOS[i+1],PPOS[i+2]),new THREE.Vector3());out.push([r,+l.x.toFixed(3),+l.y.toFixed(3),+l.z.toFixed(3),PIM[r*NCOL+4]])}return {n:SR.n,dr:ribGeo.drawRange.count,out,rest:SR.lastRest,exit:RS.exit.toArray()}};
  window.__note=()=>{note.geometry.computeBoundingBox();const b=note.geometry.boundingBox;return {t:S.storyT,u0:unwrapT0,min:b.min.toArray().map(v=>+v.toFixed(3)),max:b.max.toArray().map(v=>+v.toFixed(3)),vis:note.visible,o:noteO.toArray().map(v=>+v.toFixed(3))}};
  window.__ns=()=>{const P=NS.P;let mn=[9,9,9],mx=[-9,-9,-9];for(let k=0;k<NSN;k++)for(let c=0;c<3;c++){mn[c]=Math.min(mn[c],P[k*3+c]);mx[c]=Math.max(mx[c],P[k*3+c])}return {on:NS.on,t:+NS.t.toFixed(2),frozen:NS.frozen,mn:mn.map(v=>+v.toFixed(3)),mx:mx.map(v=>+v.toFixed(3)),st:STB.p.toArray().map(v=>+v.toFixed(3)),C:NS.C.length}};
  window.__probe=(x)=>{const o=[];for(let y=.92;y>=.6;y-=.01){let s='';for(let z=-.40;z<=-.24;z+=.005){const d=sdfAt(x,y,z);s+=d<0.0015?'#':d<.006?'+':'.'}o.push(y.toFixed(2)+' '+s)}return o.join('\n')};
  window.__probeX=(y,z0,z1)=>{let s='';for(let x=-.3;x<=.3;x+=.005){let open=true;for(let z=z0;z<=z1;z+=.003){if(sdfAt(x,y,z)<.002){open=false;break}}s+=open?'.':'#'}return s};
  window.__dbg=()=>{const lp=toLocal(CH,STB.p,new THREE.Vector3());const cp=CH.p.clone().sub(CH_REST.p);return {stoneW:STB.p.toArray().map(v=>+v.toFixed(3)),THROW,hit:ST.hit,hitV:+ST.hitV.toFixed(3),leg:ST.leg,local:[lp.x,lp.y,lp.z].map(v=>+v.toFixed(3)),unwrapT0,rollT,phase:RS.ph,U:+RS.U.toFixed(3),roll:RS.c.toArray().map(v=>+v.toFixed(2)),rows:SR.n,frozen:SR.frozenTo,chairMove:cp.toArray().map(v=>+v.toFixed(4)),com:comL.toArray().map(v=>+v.toFixed(3))}};


  /* ---------- dust and fallen plaster along the walls ---------- */
  const dr=mulberry32(515);
  function dustTex(seed,strip){const W=512,H=strip?160:384,c=canvas(W,H),x=c.getContext('2d'),r=mulberry32(seed);
    for(let i=0;i<1100;i++){const px=r()*W,py=strip?H*Math.pow(r(),1.6):r()*H,rad=3+r()*20;
      const fade=strip?Math.max(0,1-py/H)*(Math.min(1,px/60,(W-px)/60)):Math.max(0,1-Math.hypot(px/W-.5,py/H-.5)*2.1);if(fade<=0)continue;
      const g=x.createRadialGradient(px,py,0,px,py,rad),a=.075*fade*(.4+r());g.addColorStop(0,`rgba(96,90,82,${a*1.15})`);g.addColorStop(1,'rgba(96,90,82,0)');x.fillStyle=g;x.fillRect(px-rad,py-rad,rad*2,rad*2)}
    for(let i=0;i<380;i++){const px=r()*W,py=strip?H*Math.pow(r(),2.2):r()*H,f=strip?Math.max(0,1-py/H):Math.max(0,1-Math.hypot(px/W-.5,py/H-.5)*2.1);
      x.fillStyle=`rgba(${62+r()*50},${58+r()*44},${52+r()*40},${.5*r()*f})`;x.fillRect(px,py,1+r()*1.6,1+r()*1.6)}
    return c}
  const dustDecals=[];
  const addDust=(cx,cz,w,d,yaw,strip,seed)=>{const t=tex(dustTex(seed,strip),{srgb:true});t.wrapS=t.wrapT=THREE.ClampToEdgeWrapping;
    const m=new THREE.MeshStandardMaterial({map:t,transparent:true,depthWrite:false,roughness:1,envMapIntensity:.6});
    const mesh=new THREE.Mesh(new THREE.PlaneGeometry(w,d),m);mesh.rotation.set(-Math.PI/2,yaw,0,'YXZ');
    mesh.position.set(cx,.0034,cz);mesh.receiveShadow=true;mesh.renderOrder=1;scene.add(mesh);dustDecals.push({mesh,t,cx,cz})};
  // strips along the skirting (texture top edge against the wall), blobs in the corners
  const strips=[[X0+.15,.5,1.4,Math.PI/2],[X1-.15,.9,1.5,-Math.PI/2],[.9,Z1-.15,1.2,Math.PI],[-.9,Z0+.16,1.1,0]];
  strips.forEach(([x,z,len,yaw],i)=>addDust(x,z,len,.32,yaw,true,900+i));
  [[X1-.28,Z1-.28],[X0+.3,Z1-.3]].forEach(([x,z],i)=>addDust(x,z,.6,.6,dr()*6,false,950+i));
  // fallen plaster: crumbs and flakes heaped at the foot of the walls
  const crumbGeo=new THREE.IcosahedronGeometry(1,0),crumbs=[];
  const crumbMat=new THREE.MeshStandardMaterial({roughness:.97,metalness:0,flatShading:true,envMapIntensity:.5});if(TX.plasterD){const t=TX.plasterD.clone();t.needsUpdate=true;t.repeat.set(.08,.08);crumbMat.map=t}
  const clusters=[[X0,1.2,1,0,38],[X1,.55,-1,0,30],[.75,Z1,0,-1,34],[X1,1.85,-1,0,18],[X0,-.45,1,0,14]];
  const gauss=()=>{let u=0;for(let k=0;k<4;k++)u+=dr();return (u-2)/1.15};
  for(const [wx,wz,nx,nz,n] of clusters){
    const along=nx?[0,1]:[1,0];
    for(let i=0;i<n;i++){const big=dr()<.08,sz=big?.012+dr()*.02:.0018+.009*Math.pow(dr(),2.4);
      const a=gauss()*.16,out=.008+Math.abs(gauss())*(big?.05:.035);
      const x=(nx?wx+nx*(.02+out):wx+along[0]*a),z=(nz?wz+nz*(.02+out):wz+along[1]*a);
      const c=dr(),col=c<.5?[.43,.41,.38]:c<.8?[.35,.33,.3]:c<.92?[.31,.31,.29]:[.5,.48,.45];
      crumbs.push({x,z,x0:x,z0:z,s:sz,flat:big?.16+dr()*.14:.4+dr()*.45,yaw:dr()*6.28,tilt:(dr()-.5)*.5,col,vx:0,vz:0,ph:dr()*6.28})}}
  const crumbMesh=new THREE.InstancedMesh(crumbGeo,crumbMat,crumbs.length);crumbMesh.castShadow=true;crumbMesh.receiveShadow=true;crumbMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);scene.add(crumbMesh);
  const cO=new THREE.Object3D(),cC=new THREE.Color();
  function placeCrumb(c,i){const fy=floorH?floorH(c.x,c.z):0;cO.position.set(c.x,fy+c.s*c.flat*.7,c.z);cO.rotation.set(c.tilt,c.yaw,c.tilt*.5);cO.scale.set(c.s,c.s*c.flat,c.s*(.6+((c.ph*7)%1)*.6));cO.updateMatrix();crumbMesh.setMatrixAt(i,cO.matrix)}
  crumbs.forEach((c,i)=>{placeCrumb(c,i);cC.setRGB(c.col[0],c.col[1],c.col[2],THREE.SRGBColorSpace);crumbMesh.setColorAt(i,cC)});
  // a little dust lifted near the floor when the draught gusts
  const fdN=160,fdP=new Float32Array(fdN*3),fdS=[];
  for(let i=0;i<fdN;i++)fdS.push({x:X0+.2+dr()*(RW-.4),z:Z0+.15+dr()*2.6,h:.03+dr()*.4,ph:dr()*6.28});
  const fdG=new THREE.BufferGeometry();fdG.setAttribute('position',new THREE.BufferAttribute(fdP,3));
  const floorDust=new THREE.Points(fdG,new THREE.PointsMaterial({size:.0032,map:tex(dotSprite(),{srgb:true}),color:'#5f5a52',transparent:true,opacity:.0,depthWrite:false}));floorDust.frustumCulled=false;scene.add(floorDust);
  function stepDebris(dt){const t=S.time;let moved=false;
    crumbs.forEach((c,i)=>{if(c.s>.012)return;windAt(V.set(c.x,.02,c.z),W);const wm=W.length();
      if(wm>.6){const k=(wm-.6)*(.012/c.s);c.vx+=W.x*k*dt*.25;c.vz+=W.z*k*dt*.25;c.yaw+=Math.sin(t*23+c.ph)*.02*k}
      c.vx*=Math.exp(-dt*7);c.vz*=Math.exp(-dt*7);if(Math.abs(c.vx)+Math.abs(c.vz)>1e-5||wm>.6){
        c.x+=c.vx*dt;c.z+=c.vz*dt;const dx=c.x-c.x0,dz=c.z-c.z0,d=Math.hypot(dx,dz);if(d>.12){c.x=c.x0+dx/d*.12;c.z=c.z0+dz/d*.12}
        c.x=Math.min(X1-.02,Math.max(X0+.02,c.x));c.z=Math.min(Z1-.02,Math.max(Z0+.02,c.z));placeCrumb(c,i);moved=true}});
    if(moved)crumbMesh.instanceMatrix.needsUpdate=true;
    let lift=0;for(let i=0;i<fdN;i++){const f=fdS[i];windAt(V.set(f.x,.05,f.z),W);const wm=W.length(),up=Math.max(0,wm-.7);lift=Math.max(lift,up);
      fdP[i*3]=f.x+Math.sin(t*.7+f.ph)*.08*up+W.x*.05;fdP[i*3+1]=.01+f.h*Math.min(1,up*.8)*(.5+.5*Math.sin(t*1.3+f.ph));fdP[i*3+2]=f.z+Math.cos(t*.6+f.ph)*.08*up+W.z*.05}
    fdG.attributes.position.needsUpdate=true;floorDust.material.opacity=Math.min(.16,lift*.12);
    for(const d of dustDecals){windAt(V.set(d.cx,.02,d.cz),W);const wm=W.length();d.t.offset.set(Math.sin(t*.4+d.cx)*.006*wm,Math.cos(t*.33+d.cz)*.006*wm);d.mesh.material.opacity=.94+.06*Math.sin(t*.8+d.cx*3)*Math.min(1,wm)}}

  /* ---------- broken glass ---------- */
  // window glass: almost invisible faces that catch reflections, and green-tinted edges where the thickness shows
  const shardM=new THREE.MeshPhysicalMaterial({color:'#a9bbb4',transparent:true,opacity:.1,roughness:.02,metalness:0,ior:1.52,specularIntensity:.9,side:THREE.DoubleSide,depthWrite:false,envMapIntensity:.35});
  const shardE=new THREE.MeshPhysicalMaterial({color:'#456b5e',transparent:true,opacity:.72,roughness:.06,metalness:0,ior:1.52,side:THREE.DoubleSide,depthWrite:false,envMapIntensity:.9});window.__shardMats=()=>[shardM,shardE];
  const shards=[],shIv=new THREE.Matrix4(),shLp=new THREE.Vector3(),shQ=new THREE.Quaternion(),shUp=new THREE.Vector3(0,1,0),shDn=new THREE.Vector3(0,-1,0),shN=new THREE.Vector3();
  function clearShards(){shards.forEach(sh=>{sh.m.parent&&sh.m.parent.remove(sh.m);sh.m.geometry.dispose()});shards.length=0;ALLPANES.forEach(p=>p.m.visible=true);S.broken=false}
  function breakPane(pane,impact){
    if(!pane||!pane.m.visible)return;pane.m.visible=false;S.broken=true;const l={g:pane.g};
    l.g.updateMatrixWorld(true);shIv.copy(l.g.matrixWorld).invert();shLp.copy(impact).applyMatrix4(shIv);
    const w=pane.w,h=pane.h,ox=pane.cx,oy=pane.cy,cx=THREE.MathUtils.clamp(shLp.x-ox,-w/2+.04,w/2-.04),cy=THREE.MathUtils.clamp(shLp.y-oy,-h/2+.04,h/2-.04);
    const rr=mulberry32(Math.floor(S.time*1000)+7),K=13,ang=[];
    for(let i=0;i<K;i++)ang.push((i+.25+.5*rr())/K*Math.PI*2);
    const ray=a=>{const dx=Math.cos(a),dy=Math.sin(a);let tb=1e9;
      if(dx>1e-6)tb=Math.min(tb,(w/2-cx)/dx);if(dx<-1e-6)tb=Math.min(tb,(-w/2-cx)/dx);if(dy>1e-6)tb=Math.min(tb,(h/2-cy)/dy);if(dy<-1e-6)tb=Math.min(tb,(-h/2-cy)/dy);
      return {dx,dy,tb}};
    const R=ang.map(a=>{const r=ray(a),f1=.16+.14*rr(),f2=.46+.2*rr();
      return {a,I:[cx+r.dx*r.tb*f1,cy+r.dy*r.tb*f1],M:[cx+r.dx*r.tb*f2,cy+r.dy*r.tb*f2],O:[cx+r.dx*r.tb,cy+r.dy*r.tb],tb:r.tb}});
    const corners=[[w/2,h/2],[-w/2,h/2],[-w/2,-h/2],[w/2,-h/2]].map(c=>({c,a:(Math.atan2(c[1]-cy,c[0]-cx)+Math.PI*2)%(Math.PI*2)}));
    const polys=[];
    for(let i=0;i<K;i++){const A=R[i],B=R[(i+1)%K];const a0=A.a,a1=i+1<K?B.a:B.a+Math.PI*2;
      polys.push({pts:[[cx,cy],A.I,B.I],ring:0,r:0});
      polys.push({pts:[A.I,A.M,B.M,B.I],ring:1,r:.4});
      const cs=corners.filter(c=>{let ca=c.a;if(ca<a0)ca+=Math.PI*2;return ca>a0&&ca<a1}).sort((p,q)=>((p.a<a0?p.a+6.283:p.a)-(q.a<a0?q.a+6.283:q.a)));
      polys.push({pts:[A.M,A.O,...cs.map(c=>c.c),B.O,B.M],ring:2,r:.8})}
    const lq=new THREE.Quaternion();l.g.getWorldQuaternion(lq);
    for(const pl of polys){
      let gx=0,gy=0;pl.pts.forEach(p=>{gx+=p[0];gy+=p[1]});gx/=pl.pts.length;gy/=pl.pts.length;
      const shp=new THREE.Shape();pl.pts.forEach((p,j)=>j?shp.lineTo(p[0]-gx,p[1]-gy):shp.moveTo(p[0]-gx,p[1]-gy));
      const geo=new THREE.ExtrudeGeometry(shp,{depth:.0035,bevelEnabled:false});geo.translate(0,0,-.00175);
      const m=new THREE.Mesh(geo,[shardM,shardE]);m.renderOrder=2;
      const sh={m,vel:new THREE.Vector3(),ang:new THREE.Vector3(),rest:false,stuck:false,fallAt:0,target:new THREE.Quaternion(),onSill:false,ph:rr()*6.28};
      const stick=pl.ring===2&&rr()<.55;
      if(stick){m.position.set(ox+gx,oy+gy,pane.cz);l.g.add(m);sh.stuck=true;if(rr()<.45)sh.fallAt=S.time+.5+rr()*3;
        m.rotation.set((rr()-.5)*.06,(rr()-.5)*.06,(rr()-.5)*.03)}
      else{m.position.set(ox+gx,oy+gy,pane.cz).applyMatrix4(l.g.matrixWorld);m.quaternion.copy(lq);scene.add(m);
        const dist=Math.hypot(gx-cx,gy-cy)/.3,sp=(.35+1.3*Math.max(0,1-dist))*(.6+.6*rr());
        sh.vel.set(-.6,0,0).addScaledVector(stoneVel,.5).normalize().multiplyScalar(sp);
        sh.vel.x+=(rr()-.5)*.3;sh.vel.y+=.1+rr()*.5;sh.vel.z+=(rr()-.5)*.6;
        sh.ang.set((rr()-.5)*16,(rr()-.5)*16,(rr()-.5)*16)}
      m.castShadow=false;m.receiveShadow=false;shards.push(sh)}
    for(let k=0;k<28;k++){const s=.004+rr()*.012,shp=new THREE.Shape();const n=3+(rr()<.4?1:0);for(let q=0;q<n;q++){const a=q/n*6.283+rr()*.9,r=s*(.5+rr()*.7);q?shp.lineTo(Math.cos(a)*r,Math.sin(a)*r):shp.moveTo(Math.cos(a)*r,Math.sin(a)*r)}
      const geo=new THREE.ExtrudeGeometry(shp,{depth:.003,bevelEnabled:false});geo.translate(0,0,-.0015);const m=new THREE.Mesh(geo,[shardM,shardE]);m.renderOrder=2;
      m.position.set(ox+cx+(rr()-.5)*.06,oy+cy+(rr()-.5)*.06,pane.cz).applyMatrix4(l.g.matrixWorld);m.quaternion.copy(lq);scene.add(m);
      const sh={m,vel:new THREE.Vector3(),ang:new THREE.Vector3((rr()-.5)*30,(rr()-.5)*30,(rr()-.5)*30),rest:false,stuck:false,fallAt:0,target:new THREE.Quaternion(),onSill:false,ph:rr()*6.28,small:true};
      sh.vel.set(-.6,0,0).addScaledVector(stoneVel,.6).normalize().multiplyScalar(.6+rr()*1.6);sh.vel.x+=(rr()-.5)*.5;sh.vel.y+=rr()*.8;sh.vel.z+=(rr()-.5)*.9;shards.push(sh)}
  }
  window.__shards=()=>shards.map(s=>{const p=new THREE.Vector3();s.m.getWorldPosition(p);return [+p.x.toFixed(2),+p.y.toFixed(3),+p.z.toFixed(2),s.stuck?'S':s.rest?(s.onSill?'sill':'floor'):'air',s.small?1:0]});
  function landShard(sh,y,onSill){const p=sh.m.position;p.y=y;sh.rest=true;sh.onSill=onSill;sh.vel.set(sh.vel.x*.25,0,sh.vel.z*.25);
    shN.set(0,0,1).applyQuaternion(sh.m.quaternion);shQ.setFromUnitVectors(shN,shN.y>=0?shUp:shDn);sh.target.copy(shQ).multiply(sh.m.quaternion)}
  function stepShards(dt){
    for(const sh of shards){const m=sh.m,p=m.position;
      if(sh.stuck){if(sh.fallAt&&S.time>=sh.fallAt){sh.stuck=false;scene.attach(m);sh.vel.set(-.15-.2*simRand(),-.1,(simRand()-.5)*.2);sh.ang.set((simRand()-.5)*8,(simRand()-.5)*8,(simRand()-.5)*8)}else continue}
      if(!sh.rest){
        sh.vel.y-=9.8*dt;sh.vel.multiplyScalar(1-dt*.25);
        const a=sh.ang.length();if(a>1e-4){Q.setFromAxisAngle(A.copy(sh.ang).multiplyScalar(1/a),a*dt);m.quaternion.premultiply(Q)}
        p.addScaledVector(sh.vel,dt);
        if(p.x>SILL.x0&&p.x<GLX&&p.z>SILL.z0&&p.z<SILL.z1&&p.y<SILL.top+.008&&p.y>SILL.top-.12&&sh.vel.y<0){
          if(Math.abs(sh.vel.y)>1.4){p.y=SILL.top+.004;sh.vel.y*=-.18;sh.vel.x*=.5;sh.vel.z*=.5;sh.ang.multiplyScalar(.4)}else landShard(sh,SILL.top+.0016,true)}
        else{const fl=paperFloor(p.x,p.z);if(p.y<fl+.003){if(Math.abs(sh.vel.y)>1.1){p.y=fl+.003;sh.vel.y*=-.2;sh.vel.x*=.55;sh.vel.z*=.55;sh.ang.multiplyScalar(.45)}else landShard(sh,fl+.0019+simRand()*.0006,false)}}
        if(p.x<X0+.05){p.x=X0+.05;sh.vel.x*=-.2}if(p.z<Z0+.05){p.z=Z0+.05;sh.vel.z*=-.2}if(p.z>Z1-.05){p.z=Z1-.05;sh.vel.z*=-.2}
      }else{
        windAt(p,W);const wm=W.length();
        if(wm>.65){sh.vel.x+=W.x*(wm-.65)*dt*.4;sh.vel.z+=W.z*(wm-.65)*dt*.4}
        sh.vel.x*=Math.exp(-dt*6);sh.vel.z*=Math.exp(-dt*6);
        p.x=THREE.MathUtils.clamp(p.x+sh.vel.x*dt,X0+.05,sh.onSill?GLX-.03:X1-.03);p.z=THREE.MathUtils.clamp(p.z+sh.vel.z*dt,Z0+.05,Z1-.05);if(!sh.onSill&&Math.abs(sh.vel.x)+Math.abs(sh.vel.z)>1e-4)p.y=paperFloor(p.x,p.z)+.0019;
        if(sh.onSill&&p.x<SILL.x0){sh.rest=false;sh.onSill=false;sh.vel.y=0}
        if(wm>.8){Q.setFromAxisAngle(shUp,Math.sin(S.time*37+sh.ph)*.0025*(wm-.8));sh.target.premultiply(Q)}
        const sp=Math.hypot(sh.vel.x,sh.vel.z);if(sp>.004){Q.setFromAxisAngle(shUp,sp*dt*6*(sh.ph>3?1:-1));sh.target.premultiply(Q)}
        m.quaternion.slerp(sh.target,Math.min(1,dt*10))}}
  }

  /* light */
  const toSun=new THREE.Vector3(1,1.55,-1.5).normalize();
  const sun=new THREE.DirectionalLight('#ffe7c4',7.5);
  sun.target.position.set(.6,0,0);sun.position.copy(sun.target.position).addScaledVector(toSun,9);
  sun.castShadow=true;const small=Math.min(screen.width,screen.height)<700&&!STILL;sun.shadow.mapSize.set(small?2048:4096,small?2048:4096);
  Object.assign(sun.shadow.camera,{left:-6.8,right:6.8,top:6.8,bottom:-6.8,near:1,far:20});
  sun.shadow.bias=-.0002;sun.shadow.normalBias=.015;sun.shadow.radius=100+19*.022/13.6;sun.userData.ptKeep=true;scene.add(sun,sun.target);
  const hemi=new THREE.HemisphereLight('#dfe6e2','#6b5a48',.75);scene.add(hemi);
  const winLight=new THREE.RectAreaLight('#eef2ea',3.6,wW,wH);winLight.position.set(X1-.01,WC.y,WC.z);winLight.lookAt(0,WC.y,WC.z);winLight.userData.ptKeep=true;scene.add(winLight);
  // overcast daylight only gets in through the window: two broad directional lights, the walls do the masking
  const mkWin=(dir,int,tanA)=>{const l=new THREE.DirectionalLight('#eef1ec',int);l.position.copy(WC).addScaledVector(dir,-6);l.target.position.copy(WC).addScaledVector(dir,3);
    const ms=lowEnd?1024:2048;l.castShadow=true;l.shadow.mapSize.set(ms,ms);Object.assign(l.shadow.camera,{left:-6.8,right:6.8,top:6.8,bottom:-6.8,near:.5,far:16});
    l.shadow.bias=-.0004;l.shadow.normalBias=.03;l.shadow.radius=100+15.5*tanA/13.6;scene.add(l,l.target);return l};
  const skyA=mkWin(new THREE.Vector3(-1,-.34,-.4).normalize(),2.8,.2),skyB={intensity:0};
  // fit each sun/sky shadow tightly round the room (and the window wall), so its texels are a few millimetres, not a centimetre
  function fitShadow(l,tanA,k){const d=V.subVectors(l.target.position,l.position).normalize().clone(),c=new THREE.Vector3(.1,RH/2,0);
    l.target.position.copy(c);l.position.copy(c).addScaledVector(d,-9);l.updateMatrixWorld();l.target.updateMatrixWorld();l.shadow.updateMatrices(l);
    const M=l.shadow.camera.matrixWorldInverse;let hw=0;
    for(const x of [X0,X1+.45])for(const y of [0,RH])for(const z of [Z0,Z1]){const q=new THREE.Vector3(x,y,z).applyMatrix4(M);hw=Math.max(hw,Math.abs(q.x),Math.abs(q.y))}
    hw+=.08;Object.assign(l.shadow.camera,{left:-hw,right:hw,top:hw,bottom:-hw,near:1,far:20});l.shadow.camera.updateProjectionMatrix();l.shadow.radius=100+k*tanA/(2*hw);
    if(lowEnd)l.shadow.mapSize.set(2048,2048)}
  fitShadow(skyA,.2,15.5);fitShadow(sun,.022,19);
  // overcast light through the window treated as one broad source: soft shadows from everything in the room, falling away from the window
  const winSpot=new THREE.SpotLight('#eef1ec',9,0,1.05,1,1);winSpot.position.set(X1+.05,WC.y+.12,WC.z);winSpot.target.position.set(-.8,.2,.6);
  winSpot.castShadow=true;winSpot.shadow.mapSize.set(2048,2048);Object.assign(winSpot.shadow.camera,{near:.3,far:10});winSpot.shadow.bias=-.0004;winSpot.shadow.normalBias=.035;winSpot.shadow.radius=201;
  scene.add(winSpot,winSpot.target);
  const bounce=new THREE.PointLight('#ffcf9c',2.2,6,1.6);bounce.position.set(.9,.35,.6);scene.add(bounce);
  const fill=new THREE.PointLight('#e8ece8',.7,8,1.2);fill.position.set(0,2.3,1.6);scene.add(fill);

  const dustN=260,dp=new Float32Array(dustN*3),dseed=new Float32Array(dustN*3),toRoom=toSun.clone().negate();
  for(let i=0;i<dustN;i++){const z=WIN.z0+r()*(WIN.z1-WIN.z0),y=WIN.y0+r()*(WIN.y1-WIN.y0),t=r()*2.2;
    dseed[i*3]=X1+toRoom.x*t;dseed[i*3+1]=y+toRoom.y*t;dseed[i*3+2]=z+toRoom.z*t}
  const dgeo=new THREE.BufferGeometry();dgeo.setAttribute('position',new THREE.BufferAttribute(dp,3));
  const dust=new THREE.Points(dgeo,new THREE.PointsMaterial({size:.008,map:tex(dotSprite(),{srgb:true}),transparent:true,opacity:.4,depthWrite:false,blending:THREE.AdditiveBlending,color:'#fff1d8'}));
  dust.frustumCulled=false;scene.add(dust);
  function moveDust(){const t=S.time;for(let i=0;i<dustN;i++){const a=i*1.37;
    dp[i*3]=dseed[i*3]+Math.sin(t*.11+a)*.08-S.interior*((t*.05+a)%1)*.3;dp[i*3+1]=dseed[i*3+1]+Math.sin(t*.07+a*2.1)*.06+((t*.006+a)%0.3)-.15;dp[i*3+2]=dseed[i*3+2]+Math.cos(t*.09+a)*.08}
    dgeo.attributes.position.needsUpdate=true}

  // fine dust hanging in the whole room: drifts with the draught, wanders, slowly settles; you only really see it where the window light catches it
  const moteN=lowEnd?600:1400,mP=new Float32Array(moteN*3),mV=new Float32Array(moteN*3),mS=new Float32Array(moteN),mr=mulberry32(4242);
  const moteSpawn=(i,any)=>{mP[i*3]=X0+.1+mr()*(RW-.2);mP[i*3+1]=any?.05+Math.pow(mr(),.8)*(RH-.2):RH*(.35+.6*mr());mP[i*3+2]=Z0+.1+mr()*(RD-.2);mV[i*3]=mV[i*3+1]=mV[i*3+2]=0};
  for(let i=0;i<moteN;i++){moteSpawn(i,true);mS[i]=.4+Math.pow(mr(),3)*1.8}
  const mG=new THREE.BufferGeometry();mG.setAttribute('position',new THREE.BufferAttribute(mP,3));mG.setAttribute('aS',new THREE.BufferAttribute(mS,1));
  const moteMat=new THREE.ShaderMaterial({transparent:true,depthWrite:false,uniforms:{uSP:{value:winSpot.position},uSD:{value:new THREE.Vector3()},uCos:{value:Math.cos(winSpot.angle*.92)},uScale:{value:600},uOn:{value:1}},
    vertexShader:`attribute float aS;uniform vec3 uSP,uSD;uniform float uCos,uScale;varying float vL,vA;
      void main(){vec4 mv=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*mv;vec3 d=normalize(position-uSP);float c=dot(d,uSD);
        vL=smoothstep(uCos,uCos+.12,c)/(1.+.25*distance(position,uSP));float z=-mv.z;vA=smoothstep(.12,.5,z)*(1.-smoothstep(3.,6.,z));
        gl_PointSize=max(1.,aS*.0024*uScale/z);}`,
    fragmentShader:`uniform float uOn;varying float vL,vA;void main(){vec2 q=gl_PointCoord-.5;float r=dot(q,q)*4.;if(r>1.)discard;float a=(1.-r)*vA*uOn;
      vec3 c=mix(vec3(.42,.4,.37),vec3(1.,.93,.82)*1.15,vL);gl_FragColor=vec4(c,a*(.05+.7*vL));}`});
  const motes=new THREE.Points(mG,moteMat);motes.frustumCulled=false;motes.renderOrder=3;scene.add(motes);
  function moteStep(dt){moteMat.uniforms.uScale.value=renderer.domElement.height*camera.projectionMatrix.elements[5]/2;if(dt<=0)return;dt=Math.min(dt,.05);const k=1-Math.exp(-dt*1.2),t=S.time;
    for(let i=0;i<moteN;i++){const o=i*3;V.set(mP[o],mP[o+1],mP[o+2]);windAt(V,W);
      // air drift plus a little turbulence of its own; heavier specks sink faster
      const s=mS[i],tx=W.x*.22+Math.sin(t*.37+i*1.7)*.012,ty=W.y*.22-.0025*s+Math.sin(t*.29+i*2.3)*.008,tz=W.z*.22+Math.cos(t*.31+i*1.1)*.012;
      mV[o]+=(tx-mV[o])*k;mV[o+1]+=(ty-mV[o+1])*k;mV[o+2]+=(tz-mV[o+2])*k;
      mP[o]+=mV[o]*dt;mP[o+1]+=mV[o+1]*dt;mP[o+2]+=mV[o+2]*dt;
      if(mP[o+1]<.01||mP[o]<X0+.02||mP[o]>X1-.02||mP[o+2]<Z0+.02||mP[o+2]>Z1-.02||mP[o+1]>RH-.03)moteSpawn(i,false)}
    mG.attributes.position.needsUpdate=true}
  { const p0=new THREE.Vector3();winSpot.updateMatrixWorld();winSpot.target.updateMatrixWorld();moteMat.uniforms.uSD.value.copy(winSpot.target.getWorldPosition(p0)).sub(winSpot.position).normalize()}
  // light bounced off the room itself: render the room into a cube map and use it as ambient light and reflections
  const cubeRT=new THREE.WebGLCubeRenderTarget(128,{type:THREE.HalfFloatType}),cubeCam=new THREE.CubeCamera(.05,40,cubeRT),pmremG=new THREE.PMREMGenerator(renderer);
  cubeCam.position.set(.1,1.3,-.4);scene.add(cubeCam);
  function captureEnv(){const was=dust.visible;dust.visible=false;motes.visible=false;const old=scene.environment;scene.environment=null;
    cubeCam.update(renderer,scene);const env=pmremG.fromCubemap(cubeRT.texture).texture;if(old)old.dispose();scene.environment=env;dust.visible=was;motes.visible=true;
    scene.traverse(o=>{const ms=o.material?(Array.isArray(o.material)?o.material:[o.material]):[];for(const m of ms){if(m.isMeshStandardMaterial&&!m.envMap)m.envMapIntensity=.85}})}
  function setWeather(sunny){
    S.sunny=sunny;sun.visible=sunny;dust.visible=sunny;bounce.visible=sunny;
    hemi.intensity=sunny?.22:.28;winLight.intensity=sunny?2.4:2.6;winSpot.intensity=sunny?2.2:3;winLight.visible=true;skyA.intensity=sunny?3.2:4.4;skyB.intensity=sunny?1.3:1.7;fill.intensity=sunny?.35:.45;
    skyU.uExp.value=sunny?1.18:1.08;WXB.set(sunny);
  }
  // window light as set by the weather toggle, then scaled by how bright it is outside right now
  const WXB={hemi:0,winLight:0,winSpot:0,skyA:0,fill:0,sun:0,set(){this.hemi=hemi.intensity;this.winLight=winLight.intensity;this.winSpot=winSpot.intensity;this.skyA=skyA.intensity;this.fill=fill.intensity;this.sun=sun.intensity},
    apply(){const k=WX.k;hemi.intensity=this.hemi*k;winLight.intensity=this.winLight*k;winSpot.intensity=this.winSpot*k;skyA.intensity=this.skyA*k;fill.intensity=this.fill*(.6+.4*k);sun.intensity=this.sun*WX.kSun}};

  /* per-frame simulation */
  let pendingDetach=[];
  function openWindow(){
    if(!S.taped)return;S.taped=false;sfx('open',.9,false);sfx('gust',.8,true,.35);
    for(const l of leaves)l.om=2.6+simRand()*1.2;
    S.gust=2.8;S.shake=1;
    const near=sheets.filter(s=>!s.hidden).sort((a,b)=>b.prox-a.prox).slice(0,42);
    pendingDetach=[];for(let i=0;i<16;i++){const s=near[Math.floor(simRand()*near.length)];pendingDetach.push({s,at:S.time+.12+simRand()*1.8})}
    S.pushBack=1;
  }
  function tearOne(){
    const tgt=new THREE.Vector3(.35,1.42,Z0);
    const c=sheets.filter(s=>!s.hidden&&s.gap>1&&Math.abs(s.x-tgt.x)<.45&&s.y>1.3&&s.y<2.1);
    if(c.length)detach(c[Math.floor(simRand()*c.length)],.6);
  }
  // five sheets, each one that nothing else lies on, torn one after another as if the draught caught them
  function tearFive(){const c=sheets.filter(s=>!s.hidden&&s.gap>1&&s.y>.8);
    for(let k=0;k<5&&c.length;k++){const i=Math.floor(simRand()*c.length),s=c.splice(i,1)[0];pendingDetach.push({s,at:S.time+k*(.18+simRand()*.35)})}}
  function gust(){S.gust=Math.max(S.gust,1.5);sfx('gust',.85,true)}
  function resetWorld(taped){
    sheets.forEach(s=>{s.hidden=false;s.press=0});updateCover();
    flyers.forEach(f=>{if(f.wall){f.active=false;f.m.visible=false}});
    resetFloor();pendingDetach=[];S.gust=0;
    clearShards();S.storyT=-1;S.breakPending=false;storyInit();storyPose();
    S.taped=taped;S.latched=false;S.trClosed=taped;setLeaves(!taped);
    for(const l of leaves){l.tp.visible=taped;l.tp.rotation.y=0}
  }
  function stepWorld(dt){
    S.time+=dt;const t=S.time;
    const base=S.windOn?1:.18;
    wxStep(dt,STILL);WXB.apply();audioStep();
    S.exterior=base*WX.wind*(.55+.6*gustNoise(t*1.2+3))+S.gust*.3;
    const openFrac=Math.max(Math.min(1,Math.max(leafA.th,leafB.th)/.5),S.broken?.3:0,transom.th>.1?.22:0);S.openFrac=openFrac;
    S.interior=(base*WX.wind*.75*gustNoise(t*1.1)+S.gust)*(.06+.94*openFrac);
    S.gust*=Math.exp(-dt*.55);S.shake*=Math.exp(-dt*2.2);
    {let n=0;for(const f of flyers)if(f.active&&!f.rest)n++;S.flyN=Math.min(4,n)}
    // casements
    leaves.forEach((l,i)=>{
      if(S.taped||S.latched){const tg=.004+(S.taped?.012:.004)*S.exterior*(.5+.5*Math.sin(t*9+i*2))*gustNoise(t*2.7+i);l.th+=(tg-l.th)*Math.min(1,dt*(l.th>.05?2.5:20));l.om=0}
      else{
        // each sash is its own pendulum-free door: only the wind turns it. Pressure on its face (falling off as it turns edge-on to the flow),
        // air drag, a little hinge friction, a hard stop at the frame and at the plaster reveal, a soft one where it pushes into the paper
        const n=l.n||(l.n=[0,0,0]);const sub=4,hs=dt/sub;
        for(let k=0;k<sub;k++){const tt=t-dt+hs*(k+1);
          const gustL=.55+.25*Math.sin(tt*.83*l.seed+l.seed)+.18*Math.sin(tt*2.17+l.seed*3)+.12*Math.sin(tt*4.9*(1+.1*l.seed)+l.seed*7);
          const P=S.exterior*(gustL-.42)*(1+S.gust*.6);
          let tau=P*6.5*Math.max(.15,Math.cos(l.th*.85))-.9*l.om*Math.abs(l.om)-.35*l.om;
          let press=0;if(l===leafA&&l.th>l.soft){press=(l.th-l.soft)/Math.max(.02,l.max-l.soft);tau-=14*(l.th-l.soft)+2.2*l.om}
          if(Math.abs(l.om)<.02&&Math.abs(tau)<l.fric)l.om=0;else l.om+=(tau-l.fric*Math.sign(l.om||tau))/l.I*hs;
          l.th+=l.om*hs;
          if(l.th>l.max){l.th=l.max;if(l.om>.35){S.shake=Math.max(S.shake,Math.min(.3,l.om*.07));window.dispatchEvent(new CustomEvent('bang',{detail:{strength:l.om}}))}l.om=-l.om*.28}
          if(l.th<.012){l.th=.012;if(l.om<-.35)window.dispatchEvent(new CustomEvent('bang',{detail:{strength:-l.om*.7}}));l.om=-l.om*.22}
          l.pressAmt=press}
      }
      l.g.rotation.y=l.rot(l.th);
      if(!S.taped){l.tp.rotation.y=-l.side*(.5+.7*Math.abs(Math.sin(t*13+i*1.7)))*Math.min(1.2,.4+S.interior)}
      else l.tp.rotation.y=0;
    });
    { const T=transom;
      if(S.trClosed||S.taped){T.th+=(0-T.th)*Math.min(1,dt*(T.th>.05?3:20));T.om=0}
      else{const P=S.exterior*(gustNoise(t*1.9+5)-.33)*(1+S.gust*.6);T.om+=(P*9*Math.max(.15,Math.cos(T.th))-T.om*1.4-.6*T.om*Math.abs(T.om))*dt;T.th+=T.om*dt;
        const tmax=Math.min(T.max,T.absMax-leafB.th);   // it may not swing through the plaster beside the open casement
        if(T.th>tmax){T.th=tmax;if(T.om>.5)window.dispatchEvent(new CustomEvent('bang',{detail:{strength:T.om*.6}}));T.om=-Math.abs(T.om)*.3}
        if(T.th<.03){T.th=.03;T.om=Math.abs(T.om)*.3}}
      T.g.rotation.y=T.th}
    pendingDetach=pendingDetach.filter(d=>{if(t>=d.at){detach(d.s,1.6);return false}return true});
    if(S.shot===4&&!STILL&&t-(S.lastTear||0)>9&&(S.autoTears=(S.autoTears||0))<10){S.lastTear=t;S.autoTears++;tearOne()}
    stepFlyers(dt);applyPress();placeSheets();stepLamp(dt);
    if(S.storyT>=0){S.storyT+=dt;storyStep(dt)}else idleStrip(dt);storyPose();
    stepShards(dt);stepDebris(dt);moteStep(dt);
    for(const m of treeMats){m.uniforms.uTime.value=t;m.uniforms.uWind.value=.35+S.exterior*.9}
    flapU.uTime.value=t;flapU.uWind.value=S.interior;
    if(S.sunny)moveDust();
  }

  window.__lights={sun,skyA,winSpot,winLight,hemi,bounce,fill,scene};
  const storyState=()=>({t:S.storyT,p:STB.p,v:STB.v,hit:!!ST.hit,broke:!!ST.broke,contact:!!ST.contact,active:!!ST.active,roll:RS.ph,landed:!!RS.landed,rollP:RLB.p,chair:CH.p,note:NB.on,noteC:NB.c,lamp:lampG.position});window.__story=storyState;
  document.getElementById('dims').textContent=`3,6 × 5,2 × 3,2 м · ${sheets.length+floorInit.length} листов A3`;
  resetWorld(false);setWeather(false);placeSheets();storyInit();storyPose();
  return {storyState,soundOn:on=>{AU.on=on;if(on)audioStart()},setVol:v=>{AU.vol=v},lampOn:()=>lamp.on,
    wxReady:()=>WX_FRAMES?skyU.tA0.value!==BLANK:!!WX.ready[0],wxPick:n=>{WX.lock=n||null;if(WX.fade<0){cue(1-WX.slot,WX.i+1);WX.pend=true}else WX.pendAfter=true},wxLock:()=>WX.lock,wxNow:()=>{const c=WX.clip[skyU.uMix.value>.5?1:0];return c?c.n:'calm'},
    wxSeek,wxApply:()=>WXB.apply(),captureEnv,step:stepWorld,setWeather,openWindow,tearOne,tearFive,gust,resetWorld,leafA,flyers,setLamp,throwStone:()=>{AU.landed=false;storyInit();launch(THROW);S.storyT=0;S.breakPending=(S.taped||S.latched)&&S.trClosed&&!!transom.pane&&transom.pane.m.visible&&leafB.th<.1},resetStory:()=>{S.storyT=-1;S.breakPending=false;storyInit();storyPose()},prepStory:close=>{clearShards();S.storyT=-1;S.breakPending=false;storyInit();storyPose();if(close&&!S.taped){S.latched=true;S.trClosed=true;setLeaves(false);for(const l of leaves){l.tp.visible=false}}},latch:()=>{if(!S.latched)sfx('open',.65,false);S.latched=true},unlatch:()=>{if(S.latched){sfx('open',.9,false);sfx('gust',.7,true,.6)}S.latched=false;for(const l of leaves)l.om=1.6+simRand()},setTransom:c=>{S.trClosed=c;if(!c)transom.om=2}};
}

/* ---------- shots ---------- */
const SHOTS=[
  {key:'0',name:'Общий план',pos:[-.05,1.52,2.42],tgt:[.12,1.38,-1.6],fov:52},
  {key:'1',name:'Окно и листы',pos:[-.45,1.6,-1.0],tgt:[1.72,1.52,-2.42],fov:44},
  {key:'2',name:'Скотч',pos:[.98,1.74,WC.z],tgt:[1.95,1.8,WC.z],fov:46,action:'tape'},
  {key:'3',name:'Угол',pos:[-1.05,1.5,1.45],tgt:[X1,1.45,Z0],fov:50},
  {key:'4',name:'Лист',pos:[.25,1.48,-1.3],tgt:[.35,1.42,Z0],fov:40,action:'tear'},
  {key:'5',name:'Пол',pos:[.05,.24,-.75],tgt:[.65,.06,-2.45],fov:52,action:'gust'},
  {key:'6',name:'Лампочка',pos:[-1.1,1.52,.5],tgt:[-.42,1.96,-.6],fov:40,action:'gust'},
  {key:'7',name:'Камень',pos:[-.15,.62,.75],tgt:[-1.17,.26,1.72],fov:50,action:'stone'},
  // at the open window: drag to turn your head and look out, up the street, down at the roofs
  {key:'8',name:'Выглянуть в окно',pos:[X1-.34,1.6,WC.z+.04],tgt:[X1-.04,1.57,WC.z],fov:62,action:'window',look:true},
];
const controls=new OrbitControls(camera,canvasEl);
controls.enableDamping=true;controls.dampingFactor=.08;controls.minDistance=.3;controls.maxDistance=6.5;
controls.rotateSpeed=.55;controls.zoomSpeed=.7;controls.panSpeed=.6;controls.screenSpacePanning=true;
const lim={x:[X0+.12,X1-.12],y:[.12,RH-.25],z:[Z0+.15,Z1-.08]};window.__lim=lim;window.__cam=camera;window.__controls=controls;
let flying=null;
controls.addEventListener('change',()=>{if(flying)return;
  for(const v of [camera.position]){v.x=THREE.MathUtils.clamp(v.x,lim.x[0],lim.x[1]);v.y=THREE.MathUtils.clamp(v.y,lim.y[0],lim.y[1]);v.z=THREE.MathUtils.clamp(v.z,lim.z[0],lim.z[1])}});
function fovFor(shot){if(!shot)return camera.fov;const a=window.innerWidth/window.innerHeight;return a<.85?shot.fov*1.32:shot.fov}
const MOVE={KeyW:[0,1],ArrowUp:[0,1],KeyS:[0,-1],ArrowDown:[0,-1],KeyA:[-1,0],ArrowLeft:[-1,0],KeyD:[1,0],ArrowRight:[1,0]},keys=new Set();
const mvF=new THREE.Vector3(),mvR=new THREE.Vector3(),mvD=new THREE.Vector3();
function walk(dt){if(!keys.size)return;let fx=0,fz=0;keys.forEach(k=>{const m=MOVE[k];if(m){fx+=m[0];fz+=m[1]}});if(!fx&&!fz)return;
  camera.getWorldDirection(mvF);mvF.y=0;mvF.normalize();mvR.crossVectors(mvF,camera.up).normalize();
  mvD.copy(mvF).multiplyScalar(fz).addScaledVector(mvR,fx).normalize().multiplyScalar(1.3*dt);
  const nx=THREE.MathUtils.clamp(camera.position.x+mvD.x,lim.x[0],lim.x[1]),nz=THREE.MathUtils.clamp(camera.position.z+mvD.z,lim.z[0],lim.z[1]);
  mvD.set(nx-camera.position.x,0,nz-camera.position.z);camera.position.add(mvD);controls.target.add(mvD);flying=null}
function goShot(i,instant){
  const s=SHOTS[i];S.shot=i;
  const P=new THREE.Vector3(...s.pos),T=new THREE.Vector3(...s.tgt);
  if(instant){camera.position.copy(P);controls.target.copy(T);camera.fov=fovFor(s);camera.updateProjectionMatrix();flying=null}
  else flying={t:0,p0:camera.position.clone(),t0:controls.target.clone(),f0:camera.fov,P,T,F:fovFor(s)};
  controls.update();
}

/* ---------- post: grain, vignette, a slight film grade ---------- */
const GrainShader={uniforms:{tDiffuse:{value:null},uTime:{value:0},uGrain:{value:.05},uVig:{value:.3}},
  vertexShader:`varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`,
  fragmentShader:`uniform sampler2D tDiffuse;uniform float uTime,uGrain,uVig;varying vec2 vUv;
  float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
  void main(){vec2 d=vUv-.5;vec4 c=texture2D(tDiffuse,vUv);c.r=texture2D(tDiffuse,vUv-d*.0016).r;c.b=texture2D(tDiffuse,vUv+d*.0016).b;c.rgb*=1.-uVig*dot(d,d)*2.;
  float l=dot(c.rgb,vec3(.299,.587,.114));c.rgb=mix(vec3(l),c.rgb,.9);c.rgb=c.rgb*.96+.02;
  float n=h(floor(gl_FragCoord.xy)+fract(uTime*7.13)*91.7)-.5;c.rgb+=n*uGrain*(1.-l*.45);gl_FragColor=c;}`};

/* depth of field: a dense golden-angle disc with clamped highlights, so a bright bulb blurs into a soft round spot, not a ring of dots */
const DOF_FS=`#include <common>
varying vec2 vUv;uniform sampler2D tColor,tDepth;uniform float maxblur,aperture,nearClip,farClip,focus,aspect;
#include <packing>
float cocAt(vec2 uv){float vz=perspectiveDepthToViewZ(unpackRGBAToDepth(texture2D(tDepth,uv)),nearClip,farClip);return abs(clamp((focus+vz)*aperture,-maxblur,maxblur))*.42;}
vec3 tap(vec2 uv){vec3 c=texture2D(tColor,uv).rgb;float l=max(c.r,max(c.g,c.b));return c*(min(l,2.2)/max(l,1e-4));}
void main(){vec2 ac=vec2(1.0,aspect);float r0=cocAt(vUv);
  vec3 base=tap(vUv);if(r0<.0004){gl_FragColor=vec4(texture2D(tColor,vUv).rgb,1.0);return;}
  float rot=fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453)*6.2831853;
  vec3 acc=base;float ws=1.0;
  for(int i=0;i<40;i++){float fi=float(i),rr=sqrt((fi+.5)/40.0),a=fi*2.39996323+rot;vec2 o=vec2(cos(a),sin(a))*rr*r0;
    vec2 uv=vUv+o*ac;float rs=cocAt(uv);float w=clamp((rs-rr*r0)/max(r0*.25,.0003)+1.0,0.0,1.0);
    acc+=tap(uv)*w;ws+=w;}
  gl_FragColor=vec4(acc/ws,1.0);}`;

/* ---------- films: the campaign. A director moves the camera from set-up to set-up, cuts on the action,
   slows time at the moments that matter, and fires the room's events on cue. All times are scene seconds:
   they run slower inside the slow-motion stretches, so the cuts stay locked to the physics. ---------- */
const v3=(x,y,z)=>new THREE.Vector3(x,y,z);
const EASE={io:t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2,o:t=>1-Math.pow(1-t,3),i:t=>t*t*t,s:t=>t*t*(3-2*t),l:t=>t};
const fbase=w=>{w.resetWorld(false);w.resetStory();S.windOn=true;if(S.sunny)w.setWeather(false);if(w.lampOn())w.setLamp(false);w.wxPick(null)};
// shot: t:[start,end], a/b: [position, target, fov]; e: easing of the move; track: follow the stone with weight tw
const fsh=(t0,t1,a,b,o={})=>({t:[t0,t1],a:[v3(...a[0]),v3(...a[1]),a[2]],b:[v3(...(b||a)[0]),v3(...(b||a)[1]),(b||a)[2]],e:o.e||'s',track:o.track,tw:o.tw||0});
// a one-take move through key poses (Catmull-Rom)
const take=(t0,t1,keys,o={})=>({t:[t0,t1],keys:keys.map(k=>({u:k[0],p:v3(...k[1]),g:v3(...k[2]),f:k[3]})),e:o.e||'s'});
const FILMS=[
  {id:'stone',name:'Камень',sub:'Окно, полёт, удар, записка',end:13.2,
    setup:w=>{fbase(w);w.prepStory(true)},
    ev:[[1.4,w=>w.throwStone()]],
    slow:[[1.47,1.82,.22],[1.82,2.62,.38]],
    shots:[
      fsh(0,1.55,[[-.55,.95,2.35],[1.6,1.55,-1.6],40],[[-.47,.96,2.18],[1.6,1.56,-1.6],38]),
      fsh(1.55,2.0,[[.95,1.5,-.55],[1.78,1.9,-1.42],32],[[.9,1.48,-.5],[1.6,1.7,-1.25],34],{track:'stone',tw:.35}),
      fsh(2.0,2.5,[[1.3,.32,1.95],[.4,.3,.2],36],[[1.25,.3,1.95],[-.9,.15,1.4],38],{track:'stone',tw:.95}),
      fsh(2.5,4.5,[[-.62,.09,1.05],[-1.25,.14,1.76],40],[[-.69,.1,1.13],[-1.27,.12,1.74],38]),
      fsh(4.5,8.3,[[-1.66,1.32,1.36],[-1.44,.02,1.48],42],[[-1.64,.86,1.42],[-1.44,.01,1.47],38],{e:'o'}),
      fsh(8.3,13.2,[[-1.63,.62,1.5],[-1.43,.01,1.47],40],[[-1.6,.5,1.49],[-1.43,.01,1.47],35],{e:'io'})]},
  {id:'window',name:'Окно',sub:'Скотч сорван, ветер в лицо',end:12.4,
    setup:w=>{fbase(w);w.resetWorld(true)},
    ev:[[2.4,w=>w.openWindow()]],
    slow:[[2.42,3.1,.4]],
    shots:[
      fsh(0,2.42,[[1.0,1.6,-1.64],[1.95,1.47,-1.75],30],[[1.13,1.58,-1.7],[1.95,1.47,-1.75],25]),
      // the tape goes and the draught hits the lens: the camera is thrown back into the room, the leaves fly at it
      fsh(2.42,4.4,[[1.15,1.58,-1.71],[1.95,1.47,-1.75],25],[[.35,1.62,-1.45],[1.9,1.66,-1.75],44],{e:'o'}),
      fsh(4.4,7.6,[[-.72,1.25,.62],[1.6,1.55,-1.7],44],[[-.46,1.3,.3],[1.6,1.56,-1.72],42],{e:'o'}),
      fsh(7.6,12.4,[[.55,1.55,-.85],[1.8,1.6,-1.75],50],[[1.44,1.6,-1.71],[4.5,1.45,-2.3],62],{e:'io'})]},
  {id:'sheets',name:'Листы',sub:'Пять листов срывает со стены',end:12.6,
    setup:w=>{fbase(w)},
    ev:[[2.2,w=>w.tearFive()],[6.3,w=>w.gust()]],
    slow:[[2.35,3.4,.45]],
    shots:[
      fsh(0,2.45,[[-1.45,1.75,-2.24],[.8,1.55,-2.58],30],[[-1.18,1.72,-2.24],[1.0,1.5,-2.58],30]),
      fsh(2.45,6.1,[[-.95,1.0,-1.15],[.55,1.75,-2.6],46],[[-.85,1.05,-.95],[.6,1.65,-2.6],48],{e:'o'}),
      fsh(6.1,9.6,[[0,.16,-.55],[.7,.05,-2.3],50],[[.15,.14,-.78],[.72,.05,-2.32],48]),
      fsh(9.6,12.6,[[.25,2.72,-1.02],[.26,0,-1.78],56],[[.22,2.6,-1.2],[.24,0,-1.85],52])]},
  {id:'wind',name:'Сквозняк',sub:'Лампа, листы, пол',end:11.6,
    setup:w=>{fbase(w)},
    ev:[[1.0,w=>w.gust()],[5.0,w=>w.gust()],[8.6,w=>w.gust()]],
    shots:[
      fsh(0,3.3,[[-.78,1.68,.08],[-.33,1.9,-.54],30],[[-.72,1.66,.02],[-.33,1.9,-.54],27]),
      fsh(3.3,7.2,[[.6,1.45,-1.3],[-.8,1.55,-2.58],38],[[.5,1.48,-1.4],[-.6,1.55,-2.58],36]),
      fsh(7.2,11.6,[[.05,.2,-.6],[.65,.06,-2.45],48],[[0,.22,-.4],[.62,.06,-2.45],48])]},
  {id:'light',name:'Свет',sub:'Солнце в окне, лампа',end:12.4,
    setup:w=>{fbase(w)},
    ev:[[1.2,w=>{w.setWeather(true);w.captureEnv()}],[7.0,w=>{w.setWeather(false)}],[7.6,w=>{w.setLamp(true);w.captureEnv()}]],
    after:w=>{w.setLamp(false);w.captureEnv()},
    shots:[
      fsh(0,4.2,[[-1.25,1.45,1.6],[1.2,.55,-1.4],46],[[-1.1,1.4,1.3],[1.2,.55,-1.42],44]),
      fsh(4.2,7.3,[[.5,1.15,-.3],[1.6,1.6,-1.6],34],[[.6,1.2,-.45],[1.62,1.6,-1.62],32]),
      fsh(7.3,12.4,[[-1.05,1.35,.55],[-.33,1.95,-.54],34],[[-.9,1.4,.36],[-.33,1.95,-.54],30],{e:'o'})]},
  {id:'weather',name:'За окном',sub:'Ясно, дождь, гроза',end:17,
    setup:w=>{fbase(w);w.wxPick('calm')},
    ev:[[3,w=>w.wxPick('rain')],[10,w=>w.wxPick('storm')]],
    after:w=>w.wxPick(null),
    shots:[
      fsh(0,9,[[.45,1.45,-1.0],[2.6,1.65,-1.95],44],[[.82,1.47,-1.18],[2.6,1.64,-1.95],42]),
      fsh(9,17,[[.95,1.55,-1.3],[3.0,1.55,-1.9],50],[[1.1,1.56,-1.42],[3.0,1.55,-1.9],52])]},
  {id:'room',name:'Комната',sub:'Одним планом, тизер',end:17,
    setup:w=>{fbase(w)},
    shots:[take(0,17,[
      [0,[.1,1.05,2.42],[.1,1.2,-2.6],52],
      [.3,[-.6,1.42,.6],[-.2,1.5,-2.6],46],
      [.55,[.3,1.55,-.9],[1.8,1.55,-1.8],44],
      [.78,[0,1.2,-.2],[-1.25,.45,2.05],46],
      [1,[-.7,.75,1.0],[-1.3,.4,2.05],40]])]},
];
function slowAt(f,st){let s=1;for(const [a,b,k] of (f.slow||[])){const r=.1,w=sstep(a-r,a+r,st)*(1-sstep(b-r,b+r,st));s=Math.min(s,1-(1-k)*w)}return s}
function filmLength(f){let L=0;for(let st=0;st<f.end;st+=.01)L+=.01/slowAt(f,st);return L}
const DIR={on:false,f:null,list:[],st:0,ev:[],shot:-1,ts:new THREE.Vector3(),ws:1,fade:null,phase:''};
const _cr=(p0,p1,p2,p3,t,out)=>{const t2=t*t,t3=t2*t;return out.set(0,0,0).addScaledVector(p0,-.5*t3+t2-.5*t).addScaledVector(p1,1.5*t3-2.5*t2+1).addScaledVector(p2,-1.5*t3+2*t2+.5*t).addScaledVector(p3,.5*t3-.5*t2)};
const _dp=new THREE.Vector3(),_dg=new THREE.Vector3();
function filmPose(s,u){
  if(s.keys){const K=s.keys,e=EASE[s.e](u);let i=0;while(i<K.length-2&&e>K[i+1].u)i++;const a=K[i],b=K[i+1],t=(e-a.u)/Math.max(1e-6,b.u-a.u),A=K[Math.max(0,i-1)],B=K[Math.min(K.length-1,i+2)];
    _cr(A.p,a.p,b.p,B.p,t,_dp);_cr(A.g,a.g,b.g,B.g,t,_dg);return a.f+(b.f-a.f)*EASE.s(t)}
  const e=EASE[s.e](u);_dp.lerpVectors(s.a[0],s.b[0],e);_dg.lerpVectors(s.a[1],s.b[1],e);return s.a[2]+(s.b[2]-s.a[2])*e}
function filmFade(on,ms){let el=document.getElementById('filmfade');if(!el){el=document.createElement('div');el.id='filmfade';el.style.cssText='position:fixed;inset:0;background:#000;opacity:0;pointer-events:none;z-index:30;transition:opacity .6s';document.body.appendChild(el)}
  el.style.transitionDuration=(ms||600)+'ms';el.style.opacity=on?'1':'0'}
function playFilms(list){if(!list.length)return;DIR.list=list.slice();if(WALK.on)exitWalk();DIR.hooks&&DIR.hooks.start();startFilm(DIR.list.shift())}
function startFilm(f){DIR.f=f;DIR.st=0;DIR.ev=(f.ev||[]).slice().sort((a,b)=>a[0]-b[0]);DIR.shot=-1;DIR.on=true;DIR.phase='play';flying=null;S.shot=-1;
  document.body.classList.add('film');controls.enabled=false;f.setup(world);filmFade(true,0);requestAnimationFrame(()=>filmFade(false,700))}
function stopFilm(){if(!DIR.on)return;const f=DIR.f;DIR.on=false;DIR.list=[];document.body.classList.remove('film');controls.enabled=true;filmFade(false,300);
  if(f&&f.after)f.after(world);controls.target.copy(DIR.ts);DIR.hooks&&DIR.hooks.end()}
function dirStep(dt){const f=DIR.f;
  if(DIR.phase==='out'){DIR.wait-=dt;if(DIR.wait<=0){if(f.after)f.after(world);if(DIR.list.length)startFilm(DIR.list.shift());else{DIR.on=false;document.body.classList.remove('film');controls.enabled=true;filmFade(false,700);controls.target.copy(DIR.ts);DIR.hooks&&DIR.hooks.end()}}return 0}
  const ws=slowAt(f,DIR.st);DIR.ws=ws;DIR.st+=dt*ws;
  while(DIR.ev.length&&DIR.ev[0][0]<=DIR.st)DIR.ev.shift()[1](world);
  const st=DIR.st;let k=f.shots.findIndex(s=>st>=s.t[0]&&st<s.t[1]);if(k<0)k=f.shots.length-1;const s=f.shots[k];
  const fov=filmPose(s,Math.min(1,Math.max(0,(st-s.t[0])/(s.t[1]-s.t[0]))));
  if(s.track==='stone'){const p=world.storyState().p;_dg.lerp(p,s.tw)}
  if(k!==DIR.shot){DIR.shot=k;DIR.ts.copy(_dg)}else DIR.ts.lerp(_dg,1-Math.exp(-dt*9));
  camera.position.copy(_dp);const a=window.innerWidth/window.innerHeight;camera.fov=a<.85?fov*1.3:fov;camera.updateProjectionMatrix();camera.lookAt(DIR.ts);controls.target.copy(DIR.ts);
  if(st>=f.end){DIR.phase='out';DIR.wait=.75;filmFade(true,650)}
  return ws}

/* ---------- photo light: a path traced frame whenever the camera rests (computers only) ---------- */
const PHOTO={ctl:null,loading:null,failed:false,on:false,last:performance.now(),key:'',t0:performance.now(),warmed:false,
  can:()=>!PHOTO.failed&&renderer.capabilities.isWebGL2&&!!renderer.extensions.get('EXT_color_buffer_float')};
window.__PHOTO=PHOTO;
function photoPoke(){PHOTO.last=performance.now();if(PHOTO.ctl&&PHOTO.ctl.active)PHOTO.ctl.stop()}
['pointerdown','wheel','keydown'].forEach(e=>window.addEventListener(e,photoPoke,{passive:true,capture:true}));
// the live frame dissolves into the traced one: a copy of the last live frame laid over the canvas, faded out
function photoFade(){const c=document.getElementById('ptfade');if(!c)return;const src=renderer.domElement;
  if(c.width!==src.width||c.height!==src.height){c.width=src.width;c.height=src.height}
  c.getContext('2d').drawImage(src,0,0);c.style.transition='none';c.style.opacity='1';
  requestAnimationFrame(()=>requestAnimationFrame(()=>{c.style.transition='opacity 1.4s ease';c.style.opacity='0'}))}
function photoLoad(){if(PHOTO.loading)return PHOTO.loading;
  PHOTO.loading=import('./photo.js').then(m=>{PHOTO.ctl=m.createPhoto({renderer,scene,view,grainShader:GrainShader,textureSize:HQ?2048:1024,
      makeBokeh:()=>{const b=new BokehPass(scene,view,{focus:2,aperture:.0016,maxblur:.007});b.materialBokeh.fragmentShader=DOF_FS;b.materialBokeh.needsUpdate=true;b.enabled=bokeh?bokeh.enabled:true;return b},
      maxPixels:HQ?2.4e6:7e5,onProgress:v=>{PHOTO.progress&&PHOTO.progress(v)}});
    PHOTO.ctl.setSize(window.innerWidth,window.innerHeight);return PHOTO.ctl})
    .catch(e=>{console.warn('photo light unavailable',e);PHOTO.failed=true;return null});
  return PHOTO.loading}

/* ---------- boot ---------- */
let composer,gtao,grain,world,bokeh;
function focusDOF(){if(bokeh)bokeh.uniforms.focus.value=S.shot>=0&&SHOTS[S.shot].look?40:view.position.distanceTo(controls.target)}
function resize(){
  const w=window.innerWidth,h=window.innerHeight;renderer.setSize(w,h,false);
  for(const c of [camera,view]){c.aspect=w/h}camera.fov=fovFor(SHOTS[S.shot]);camera.updateProjectionMatrix();
  if(composer)composer.setSize(w,h);
  if(PHOTO.ctl)PHOTO.ctl.setSize(w,h);
}
window.addEventListener('resize',resize);
const reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function syncView(){
  view.position.copy(camera.position);view.quaternion.copy(camera.quaternion);view.fov=camera.fov;view.aspect=camera.aspect;
  const t=S.time,k=reduce?0:1;
  view.position.x+=Math.sin(t*.31)*.004*k;view.position.y+=Math.sin(t*.43+1)*.003*k;
  const sh=S.shake*k;
  if(sh>.001){view.position.x+=(Math.sin(t*37)+Math.sin(t*23.3))*.006*sh;view.position.y+=(Math.sin(t*41.7)+Math.sin(t*19.1))*.005*sh;
    view.rotateZ(Math.sin(t*29)*.006*sh);view.rotateX(Math.sin(t*33.3)*.004*sh)}
  view.updateProjectionMatrix();
}

const $=id=>document.getElementById(id);
const ui={wind:$('wind'),weather:$('weather'),shots:$('shots'),act:$('bAct'),now:$('now')};
const ACTS={
  tape:()=>S.taped?{label:'Сорвать скотч',run:()=>world.openWindow()}:{label:'Заклеить окно',run:()=>world.resetWorld(true)},
  window:()=>(S.taped||S.latched)?{label:'Открыть окно',run:()=>S.taped?world.openWindow():world.unlatch()}:{label:'Закрыть окно',run:()=>world.latch()},
  stone:()=>({label:S.storyT<0?'Бросить камень':'Ещё раз',run:()=>{world.prepStory(S.shot===7);world.throwStone()}}),
  tear:()=>({label:'Сорвать листы',run:()=>world.tearFive()}),
  gust:()=>({label:'Порыв ветра',run:()=>world.gust()}),
  reset:()=>({label:'Сбросить',run:()=>world.resetWorld(false)}),
};
function actionList(){if(S.shot<0)return [];const a=SHOTS[S.shot].action;return a?[a==='tape'?'tape':a]:[]}
let syncUI=()=>{};
function refreshAction(){
  window.__syncTr&&window.__syncTr();
  const l=actionList(),a=l.length?ACTS[l[0]]():null;
  ui.act.hidden=!a||DIR.on;if(a){ui.act.textContent=a.label;ui.act.onclick=()=>{a.run();refreshAction()}}
  window.__isleLayout&&window.__isleLayout();
  syncUI();
}

/* walking: look by dragging, step with the stick (or WASD), height on the right (or Q / E); smooth, never jerky */
const WALK={on:false,yaw:0,pitch:0,yawT:0,pitchT:0,joy:{x:0,y:0},hT:1.6,vel:new THREE.Vector3(),ptr:new Map(),pinch:0,sens:1};
const _wf=new THREE.Vector3(),_wr=new THREE.Vector3(),_wv=new THREE.Vector3(),_wl=new THREE.Vector3();
function enterWalk(){if(WALK.on)return;flying=null;S.shot=-1;
  const d=_wl.subVectors(controls.target,camera.position).normalize();
  WALK.yaw=WALK.yawT=Math.atan2(d.x,d.z);WALK.pitch=WALK.pitchT=Math.asin(THREE.MathUtils.clamp(d.y,-1,1))*.6;WALK.hT=camera.position.y;WALK.vel.set(0,0,0);
  WALK.on=true;controls.enabled=false;document.body.classList.add('walking');refreshAction();walkHUD()}
function exitWalk(){if(!WALK.on)return;WALK.on=false;
  _wl.set(Math.sin(WALK.yaw)*Math.cos(WALK.pitch),Math.sin(WALK.pitch),Math.cos(WALK.yaw)*Math.cos(WALK.pitch));controls.target.copy(camera.position).addScaledVector(_wl,1.2);
  controls.enabled=true;document.body.classList.remove('walking');WALK.joy.x=WALK.joy.y=0;WALK.ptr.clear()}
function walkStep(dt){let fx=WALK.joy.x,fz=WALK.joy.y,fy=0,turn=0;
  keys.forEach(k=>{if(k==='ArrowLeft')turn+=1;else if(k==='ArrowRight')turn-=1;else{const m=MOVE[k];if(m){fx+=m[0];fz+=m[1]}}if(k==='KeyE')fy+=1;if(k==='KeyQ')fy-=1});
  WALK.yawT+=turn*dt*1.5*WALK.sens;
  WALK.hT=THREE.MathUtils.clamp(WALK.hT+fy*dt*.9,lim.y[0]+.1,lim.y[1]);
  _wf.set(Math.sin(WALK.yaw),0,Math.cos(WALK.yaw));_wr.set(-Math.cos(WALK.yaw),0,Math.sin(WALK.yaw));
  _wv.copy(_wf).multiplyScalar(fz).addScaledVector(_wr,fx);const l=_wv.length();if(l>1)_wv.multiplyScalar(1/l);_wv.multiplyScalar(1.35*WALK.sens);
  WALK.vel.lerp(_wv,1-Math.exp(-dt*6));
  const p=camera.position;p.addScaledVector(WALK.vel,dt);p.y+=(WALK.hT-p.y)*(1-Math.exp(-dt*5));
  p.x=THREE.MathUtils.clamp(p.x,lim.x[0],lim.x[1]);p.z=THREE.MathUtils.clamp(p.z,lim.z[0],lim.z[1]);p.y=THREE.MathUtils.clamp(p.y,lim.y[0],lim.y[1]);
  const k=1-Math.exp(-dt*14);WALK.yaw+=(WALK.yawT-WALK.yaw)*k;WALK.pitch+=(WALK.pitchT-WALK.pitch)*k;
  _wl.set(Math.sin(WALK.yaw)*Math.cos(WALK.pitch),Math.sin(WALK.pitch),Math.cos(WALK.yaw)*Math.cos(WALK.pitch)).add(p);camera.lookAt(_wl);
  walkHUD()}
let walkHUD=()=>{};window.__walk=WALK;

function selectShot(i,instant){
  const prev=S.shot;if(i>=0)exitWalk();
  if(i>=0)goShot(i,instant);else{S.shot=-1;flying=null}
  const act=i>=0?SHOTS[i].action:null;
  if(act==='tape')world.resetWorld(true);
  else if(i>=0&&SHOTS[i].look){if(S.taped)world.openWindow();else world.unlatch()}
  else if(act==='stone')world.prepStory(true);
  else if(i>=0&&prev===2&&S.taped)world.resetWorld(false);
  refreshAction();
}
function doAction(){const l=actionList();if(l.length){ACTS[l[0]]().run();refreshAction()}}

build().then(async w=>{
  world=w;window.__world=w;
  const startShot=params.has('shot')?Number(params.get('shot')):3;
  resize();
  composer=new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene,view));
  gtao=new GTAOPass(scene,view,window.innerWidth,window.innerHeight);
  gtao.updateGtaoMaterial({radius:.34,distanceExponent:1.3,thickness:.8,scale:1.35,samples:16});
  gtao.updatePdMaterial({lumaPhi:10,depthPhi:4,normalPhi:4,radius:4,rings:2,samples:16});
  gtao.blendIntensity=.88;
  if(!params.has('noao'))composer.addPass(gtao);
  bokeh=new BokehPass(scene,view,{focus:2,aperture:.0016,maxblur:.007});bokeh.materialBokeh.fragmentShader=DOF_FS;bokeh.materialBokeh.needsUpdate=true;bokeh.enabled=!(Math.min(screen.width,screen.height)<700&&!STILL)&&!params.has('nodof');composer.addPass(bokeh);
  composer.addPass(new OutputPass());
  grain=new ShaderPass(GrainShader);composer.addPass(grain);
  resize();
  if(params.get('weather')==='sun'){world.setWeather(true);ui.weather.setAttribute('aria-pressed','true')}
  selectShot(startShot,true);
  world.captureEnv();
  if(params.has('cam')){const c=params.get('cam').split(',').map(Number);camera.position.set(c[0],c[1],c[2]);controls.target.set(c[3],c[4],c[5]);if(c[6])camera.fov=c[6];camera.updateProjectionMatrix();controls.update()}

  /* ---------- interface: one liquid-glass island ---------- */
  const store={get:k=>{try{return localStorage.getItem(k)}catch(e){return null}},set:(k,v)=>{try{localStorage.setItem(k,v)}catch(e){}}};
  const isle=$('isle'),isleBar=$('isleBar'),isleBody=$('isleBody'),hint=$('hint'),corners=$('corners');
  const isleIn=document.createElement('div');isleIn.className='isle-in';isle.insertBefore(isleIn,isleBar);isleIn.appendChild(isleBar);isleIn.appendChild(isleBody);
  const secs=[...isleBody.querySelectorAll('.sec')],secBtns=[...isleBar.querySelectorAll('.ib[data-sec]')];
  const DOCKS=['b','t','l','r','bl','br','tl','tr'];
  const ISL={dock:DOCKS.includes(store.get('room.dock'))?store.get('room.dock'):'b',open:null,dot:store.get('room.dot')||null,drag:null};
  // the safe-area insets, read once from CSS
  const safe=(()=>{const d=document.createElement('div');d.style.cssText='position:fixed;visibility:hidden;padding:env(safe-area-inset-top,0px) env(safe-area-inset-right,0px) env(safe-area-inset-bottom,0px) env(safe-area-inset-left,0px)';document.body.appendChild(d);const c=getComputedStyle(d),r={t:parseFloat(c.paddingTop)||0,r:parseFloat(c.paddingRight)||0,b:parseFloat(c.paddingBottom)||0,l:parseFloat(c.paddingLeft)||0};d.remove();return r})();
  const M=14,DOT=18;
  function isleSize(){if(ISL.dot)return [DOT,DOT];return [Math.ceil(isleIn.offsetWidth),Math.ceil(isleIn.offsetHeight)]}
  function islePos(dock,w,h){const W=innerWidth,H=innerHeight,l=M+safe.l,r=W-w-M-safe.r,t=M+safe.t,b=H-h-M-safe.b,cx=(W-w)/2,cy=(H-h)/2;
    return {b:[cx,b],t:[cx,t],l:[l,cy],r:[r,cy],bl:[l,b],br:[r,b],tl:[l,t],tr:[r,t]}[dock]}
  function isleLayout(){
    isle.dataset.dock=ISL.dot||ISL.dock;isle.classList.toggle('dot',!!ISL.dot);
    const [w,h]=isleSize(),[x,y]=ISL.dot?(()=>{const W=innerWidth,H=innerHeight,c=ISL.dot;return [c[1]==='l'?10+safe.l:W-DOT-10-safe.r,c[0]==='t'?10+safe.t:H-DOT-10-safe.b]})():islePos(ISL.dock,w,h);
    isle.style.width=w+'px';isle.style.height=h+'px';isle.style.transform=`translate(${Math.round(x)}px,${Math.round(y)}px)`;
    isle.style.borderRadius=ISL.dot?'50%':(ISL.open?'28px':Math.min(30,h/2)+'px');
    const low=!ISL.dot&&ISL.dock[0]==='b';document.documentElement.style.setProperty('--wb',(low?h+M+12:24)+'px');
    hint.classList.toggle('low',!ISL.dot&&ISL.dock[0]==='t')}
  window.__isleLayout=isleLayout;
  // first placement without animation
  isle.classList.add('notrans');isleLayout();requestAnimationFrame(()=>requestAnimationFrame(()=>isle.classList.remove('notrans')));
  addEventListener('resize',()=>{isle.classList.add('notrans');isleLayout();requestAnimationFrame(()=>isle.classList.remove('notrans'))});
  if(window.ResizeObserver)new ResizeObserver(()=>{if(!ISL.drag)isleLayout()}).observe(isleIn);

  function openSec(id){
    const prev=ISL.open;if(prev===id)id=null;ISL.open=id;
    secs.forEach(s=>{const on=s.dataset.sec===id;if(s.classList.contains('show')&&!on){s.classList.remove('show')}s.classList.toggle('show',on)});
    secBtns.forEach(b=>b.classList.toggle('open',b.dataset.sec===id));
    showHint(id==='mode');isleLayout();syncUI()}
  const closeSecs=()=>{if(ISL.open)openSec(null)};
  function fold(corner){ISL.dot=corner;store.set('room.dot',corner);closeSecs();isleLayout()}
  function unfold(){const c=ISL.dot;ISL.dot=null;store.set('room.dot','');if(['tl','tr','bl','br'].includes(c)){ISL.dock=c;store.set('room.dock',c)}isleLayout()}
  const nearestCorner=()=>{const r=isle.getBoundingClientRect(),cx=r.left+r.width/2,cy=r.top+r.height/2;return (cy<innerHeight/2?'t':'b')+(cx<innerWidth/2?'l':'r')};
  if(ISL.dot&&!['tl','tr','bl','br'].includes(ISL.dot))ISL.dot=null;isleLayout();

  // dragging: anywhere on the island; past a few pixels it is a drag, not a tap. Into a corner = fold to a dot
  let suppressClick=false;
  isle.addEventListener('pointerdown',e=>{if(e.target.closest('input'))return;const r=isle.getBoundingClientRect();ISL.drag={id:e.pointerId,x0:e.clientX,y0:e.clientY,ox:e.clientX-r.left,oy:e.clientY-r.top,on:false,dot:!!ISL.dot}});
  window.addEventListener('pointermove',e=>{const d=ISL.drag;if(!d||d.id!==e.pointerId)return;
    if(!d.on){if(Math.hypot(e.clientX-d.x0,e.clientY-d.y0)<8)return;d.on=true;try{isle.setPointerCapture(e.pointerId)}catch(_){}if(ISL.open)openSec(null);if(ISL.dot){ISL.dot=null;isleLayout()}isle.classList.add('drag');corners.classList.add('on')}
    const [w,h]=isleSize();const x=Math.max(4,Math.min(innerWidth-w-4,e.clientX-Math.min(d.ox,w-10))),y=Math.max(4,Math.min(innerHeight-h-4,e.clientY-Math.min(d.oy,h-10)));
    isle.style.transform=`translate(${x}px,${y}px)`;
    const c=hotCorner(e.clientX,e.clientY);corners.querySelectorAll('i').forEach(i=>i.classList.toggle('hot',i.dataset.c===c));d.corner=c;d.cx=e.clientX;d.cy=e.clientY});
  const endDrag=e=>{const d=ISL.drag;if(!d||d.id!==e.pointerId)return;ISL.drag=null;if(!d.on)return;suppressClick=true;setTimeout(()=>suppressClick=false,60);
    isle.classList.remove('drag');corners.classList.remove('on');corners.querySelectorAll('i').forEach(i=>i.classList.remove('hot'));
    if(d.corner){fold(d.corner);return}
    // the nearest of the eight places
    const [w,h]=isleSize();let best='b',bd=1e9;for(const k of DOCKS){const [x,y]=islePos(k,w,h),dd=Math.hypot(x+w/2-d.cx,y+h/2-d.cy);if(dd<bd){bd=dd;best=k}}
    ISL.dock=best;store.set('room.dock',best);isleLayout()};
  window.addEventListener('pointerup',endDrag);window.addEventListener('pointercancel',endDrag);
  isle.addEventListener('click',e=>{if(suppressClick){e.stopPropagation();e.preventDefault();return}if(ISL.dot){e.stopPropagation();unfold()}},true);
  function hotCorner(x,y){const R=70,W=innerWidth,H=innerHeight;for(const [c,cx,cy] of [['tl',0,0],['tr',W,0],['bl',0,H],['br',W,H]])if(Math.hypot(x-cx,y-cy)<R)return c;return null}

  secBtns.forEach(b=>b.onclick=()=>openSec(b.dataset.sec));
  canvasEl.addEventListener('pointerdown',()=>{closeSecs();stopTour()});

  // hint at the far edge while the modes are open
  const HINTS={live:'<b>Лайв.</b> Комната живёт: ветер, погода за окном, свет. Крутите, ходите, бросайте камень.',
    photo:'<b>Фото.</b> Остановите камеру — свет просчитается как на снимке, за пару секунд. Любое касание вернёт движение.',
    photoNo:'<b>Фото</b> здесь недоступно: браузер не умеет считать свет по-настоящему. Попробуйте свежий Chrome или Safari.'};
  function showHint(on){hint.classList.toggle('show',!!on);if(on)hint.innerHTML=HINTS[MODE.v==='photo'?'photo':'live']}

  // modes: live / photo
  const MODE={v:'live'};const canPhoto=PHOTO.can()&&!STILL;
  const modeBtns=[...document.querySelectorAll('#modeSeg button')];
  if(!canPhoto)modeBtns[1].disabled=true;
  function setMode(v,silent){if(v==='photo'&&!canPhoto){hint.innerHTML=HINTS.photoNo;hint.classList.add('show');return}
    MODE.v=v;store.set('room.mode',v);PHOTO.on=v==='photo';if(!PHOTO.on)photoPoke();
    modeBtns.forEach(b=>b.classList.toggle('on',b.dataset.mode===v));$('modeHalf').style.opacity=v==='photo'?0:1;
    if(!silent&&ISL.open==='mode')hint.innerHTML=HINTS[v]}
  modeBtns.forEach(b=>b.onclick=()=>setMode(b.dataset.mode));
  {const sv=store.get('room.mode');setMode(params.has('pt')?'photo':params.has('nopt')?'live':sv==='photo'&&canPhoto?'photo':'live',true)}
  // the photo light's progress runs round the mode icon
  PHOTO.progress=v=>{const b=secBtns[0],r=$('ptRing');if(v<0){b.classList.remove('dev');r.style.strokeDashoffset=1}else{b.classList.toggle('dev',v<1);r.style.strokeDashoffset=String(1-Math.min(1,v))}};

  // views
  SHOTS.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.className='chip';b.innerHTML=`<small>${s.key}</small>${s.name}`;b.onclick=()=>{stopTour();selectShot(i)};ui.shots.appendChild(b)});
  const toggleWalk=()=>{stopTour();if(WALK.on){exitWalk();selectShot(SHOTS.findIndex(s=>s.key==='0'))}else{closeSecs();enterWalk()}syncUI()};
  $('bWalk').onclick=toggleWalk;
  const TOUR=[0,1,3,6,4,5,7,8];let tourT=null;
  function stopTour(){if(tourT){clearInterval(tourT);tourT=null;syncUI()}}
  $('bTour').onclick=()=>{if(tourT){stopTour();return}closeSecs();let k=Math.max(0,TOUR.indexOf(S.shot));const go=()=>{k=(k+1)%TOUR.length;selectShot(TOUR[k])};go();tourT=setInterval(go,9000);syncUI()};

  // films
  FILMS.forEach(f=>{const b=document.createElement('button');b.type='button';b.className='film';b.innerHTML=`<i></i><span><b>${f.name}</b><small>${Math.round(filmLength(f))} с · ${f.sub}</small></span>`;b.title=f.sub;
    b.onclick=()=>{closeSecs();playFilms([f])};$('films').appendChild(b)});
  $('bReel').onclick=()=>{closeSecs();playFilms(FILMS.slice())};

  // scene
  const winBtn=$('bWin'),trBtn=$('transom'),tapeBtn=$('bTape'),lampBtn=$('lamp'),sb=$('sound');
  winBtn.onclick=()=>{if(S.taped)world.openWindow();else if(S.latched)world.unlatch();else world.latch();syncUI()};
  trBtn.onclick=()=>{world.setTransom(!S.trClosed);syncUI()};
  tapeBtn.onclick=()=>{if(S.taped)world.openWindow();else world.resetWorld(true);syncUI()};
  ui.wind.onclick=()=>{S.windOn=!S.windOn;syncUI()};
  $('bGust').onclick=()=>world.gust();
  ui.weather.onclick=()=>{world.setWeather(!S.sunny);world.captureEnv();syncUI()};
  lampBtn.onclick=()=>{world.setLamp(!world.lampOn());world.captureEnv();syncUI()};
  if(params.get('lamp')==='1')world.setLamp(true);
  const WXN=[[null,'Сам'],['calm','Ясно'],['wind','Ветер'],['shade','Пасмурно'],['rain','Дождь'],['storm','Гроза']];
  const wxBtns=WXN.map(([n,t])=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=t;b.onclick=()=>{world.wxPick(n);syncUI()};$('wxChips').appendChild(b);return [n,b]});
  $('bStone').onclick=()=>{closeSecs();stopTour();selectShot(SHOTS.findIndex(s=>s.action==='stone'));setTimeout(()=>{world.prepStory(true);world.throwStone();refreshAction()},1400)};
  $('bTear').onclick=()=>{world.tearFive()};
  $('bReset').onclick=()=>{world.resetWorld(false);world.resetStory();syncUI()};
  window.__syncTr=()=>syncUI();

  // sound
  const vol=$('vol');{const v=store.get('room.vol');if(v!==null)vol.value=v}world.setVol(+vol.value);
  let sndOn=true;vol.oninput=()=>{world.setVol(+vol.value);store.set('room.vol',vol.value);if(+vol.value>0&&!sndOn){sndOn=true;world.soundOn(true)}syncUI()};
  sb.onclick=()=>{sndOn=!sndOn;world.soundOn(sndOn);syncUI()};
  // screen
  const bFull=$('bFull'),fsOK=!!(document.fullscreenEnabled||document.webkitFullscreenEnabled);bFull.hidden=!fsOK;
  bFull.onclick=()=>{const d=document,el=d.documentElement;if(d.fullscreenElement||d.webkitFullscreenElement)(d.exitFullscreen||d.webkitExitFullscreen).call(d);else (el.requestFullscreen||el.webkitRequestFullscreen).call(el)};
  document.addEventListener('fullscreenchange',()=>syncUI());
  $('bDot').onclick=()=>fold(nearestCorner());
  const bShot=$('bShot');bShot.hidden=true;let dls=null;
  const inClaude=!!(window.claude&&window.claude.use);
  /* on claude.ai the still goes through the downloads capability; on your own site it is an ordinary file download */
  if(!inClaude){dls={save:({filename,data})=>new Promise(r=>{const u=URL.createObjectURL(data),a=document.createElement('a');a.href=u;a.download=filename;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(u),4000);r({status:'saved'})})};bShot.hidden=false}
  else (async()=>{try{dls=await window.claude.use('downloads')}catch(e){dls=null}bShot.hidden=!dls})();
  bShot.onclick=()=>{if(!dls)return;if(PHOTO.ctl&&PHOTO.ctl.shown)PHOTO.ctl.step({focus:bokeh.uniforms.focus.value,time:S.time});else{if(!WALK.on)controls.update();syncView();composer.render()}canvasEl.toBlob(async b=>{if(!b)return;try{await dls.save({filename:'eshche-ne-konets.png',data:b})}catch(e){if(e&&['unavailable','not_granted','capability_disabled','capability_removed'].includes(e.code))bShot.hidden=true}},'image/png')};
  // quality
  const dpr=window.devicePixelRatio||1,small=Math.min(screen.width,screen.height)<700;
  const setQ=q=>{renderer.setPixelRatio(q==='hi'?Math.min(dpr,2):q==='lo'?1:Math.min(dpr,1.5));gtao.enabled=q!=='lo'&&!params.has('noao');bokeh.enabled=q==='hi'||(q==='mid'&&!small);
    if(PHOTO.ctl)PHOTO.ctl.setBokeh(bokeh.enabled);resize();store.set('room.q',q);document.querySelectorAll('#qChips button').forEach(b=>b.classList.toggle('on',b.dataset.q===q))};
  document.querySelectorAll('#qChips button').forEach(b=>b.onclick=()=>setQ(b.dataset.q));
  setQ(store.get('room.q')||(HQ?'hi':'mid'));
  // camera speed
  const sens=$('sens');{const v=store.get('room.sens');if(v!==null)sens.value=v}
  const setSens=()=>{const k=+sens.value;WALK.sens=k;controls.rotateSpeed=.55*k;controls.zoomSpeed=.7*k;controls.panSpeed=.6*k;store.set('room.sens',sens.value)};sens.oninput=setSens;setSens();

  // the walking controls: stick and height
  { const joy=$('joy'),knob=joy.querySelector('i');let jid=null,c0=null;
    const R=()=>joy.clientWidth/2-10;
    const set=(x,y)=>{const r=R(),l=Math.hypot(x,y),k=l>r?r/l:1;x*=k;y*=k;knob.style.transform=`translate(${x}px,${y}px)`;WALK.joy.x=x/r;WALK.joy.y=-y/r};
    joy.addEventListener('pointerdown',e=>{jid=e.pointerId;joy.setPointerCapture(jid);joy.classList.add('drag');const b=joy.getBoundingClientRect();c0=[b.left+b.width/2,b.top+b.height/2];set(e.clientX-c0[0],e.clientY-c0[1]);e.stopPropagation()});
    joy.addEventListener('pointermove',e=>{if(e.pointerId!==jid)return;set(e.clientX-c0[0],e.clientY-c0[1])});
    const up=e=>{if(e.pointerId!==jid)return;jid=null;joy.classList.remove('drag');set(0,0)};joy.addEventListener('pointerup',up);joy.addEventListener('pointercancel',up);
    const hg=$('hgt'),hk=hg.querySelector('i'),hl=$('hgtLbl');let hid=null;const H0=lim.y[0]+.1,H1=lim.y[1];
    const fromY=y=>{const b=hg.getBoundingClientRect(),t=1-THREE.MathUtils.clamp((y-b.top-22)/(b.height-44),0,1);WALK.hT=H0+(H1-H0)*t};
    hg.addEventListener('pointerdown',e=>{hid=e.pointerId;hg.setPointerCapture(hid);fromY(e.clientY);e.stopPropagation()});
    hg.addEventListener('pointermove',e=>{if(e.pointerId===hid)fromY(e.clientY)});
    const hup=e=>{if(e.pointerId===hid)hid=null};hg.addEventListener('pointerup',hup);hg.addEventListener('pointercancel',hup);
    walkHUD=()=>{const t=(camera.position.y-H0)/(H1-H0),h=hg.clientHeight;hk.style.top=(h-39-THREE.MathUtils.clamp(t,0,1)*(h-44)+2)+'px';hl.textContent=camera.position.y.toFixed(1).replace('.',',')+' м'};
    // looking about by dragging on the room; two fingers pinch to step forward or back; the wheel steps too
    canvasEl.addEventListener('pointerdown',e=>{if(!WALK.on)return;WALK.ptr.set(e.pointerId,[e.clientX,e.clientY]);canvasEl.setPointerCapture(e.pointerId);if(WALK.ptr.size===2){const [a,b]=[...WALK.ptr.values()];WALK.pinch=Math.hypot(a[0]-b[0],a[1]-b[1])}});
    canvasEl.addEventListener('pointermove',e=>{if(!WALK.on||!WALK.ptr.has(e.pointerId))return;const p=WALK.ptr.get(e.pointerId),dx=e.clientX-p[0],dy=e.clientY-p[1];WALK.ptr.set(e.pointerId,[e.clientX,e.clientY]);
      if(WALK.ptr.size===1){const k=.0042*WALK.sens;WALK.yawT+=dx*k;WALK.pitchT=THREE.MathUtils.clamp(WALK.pitchT+dy*k,-1.3,1.3)}
      else if(WALK.ptr.size===2){const [a,b]=[...WALK.ptr.values()],d=Math.hypot(a[0]-b[0],a[1]-b[1]);const s=(d-WALK.pinch)*.004;WALK.pinch=d;
        camera.position.x=THREE.MathUtils.clamp(camera.position.x+Math.sin(WALK.yaw)*s,lim.x[0],lim.x[1]);camera.position.z=THREE.MathUtils.clamp(camera.position.z+Math.cos(WALK.yaw)*s,lim.z[0],lim.z[1])}});
    const pup=e=>{WALK.ptr.delete(e.pointerId)};canvasEl.addEventListener('pointerup',pup);canvasEl.addEventListener('pointercancel',pup);
    canvasEl.addEventListener('wheel',e=>{if(!WALK.on)return;e.preventDefault();const s=-e.deltaY*.0012*WALK.sens;
      camera.position.x=THREE.MathUtils.clamp(camera.position.x+Math.sin(WALK.yaw)*s,lim.x[0],lim.x[1]);camera.position.z=THREE.MathUtils.clamp(camera.position.z+Math.cos(WALK.yaw)*s,lim.z[0],lim.z[1])},{passive:false})}

  // state of everything shown in the menus
  syncUI=()=>{
    [...ui.shots.children].forEach((b,j)=>b.classList.toggle('on',j===S.shot&&!WALK.on));
    $('bWalk').classList.toggle('on',WALK.on);$('bTour').classList.toggle('on',!!tourT);
    winBtn.textContent=(S.taped||S.latched)?'Открыть':'Закрыть';
    trBtn.setAttribute('aria-pressed',String(!S.trClosed));
    tapeBtn.textContent=S.taped?'Сорвать скотч':'Заклеить скотчем';
    ui.wind.setAttribute('aria-pressed',String(S.windOn));ui.weather.setAttribute('aria-pressed',String(S.sunny));lampBtn.setAttribute('aria-pressed',String(world.lampOn()));
    const lk=world.wxLock();wxBtns.forEach(([n,b])=>b.classList.toggle('on',n===lk));
    sb.setAttribute('aria-pressed',String(sndOn));
    bFull.textContent=(document.fullscreenElement||document.webkitFullscreenElement)?'Обычный экран':'Во весь экран'};
  setInterval(syncUI,700);
  DIR.hooks={start:()=>{closeSecs();stopTour();photoPoke()},end:()=>{refreshAction();syncUI()}};

  window.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;
    if(e.key==='Escape'){if(DIR.on)stopFilm();closeSecs();return}
    if(DIR.on)return;
    if(/^[0-8]$/.test(e.key)){const k=SHOTS.findIndex(s=>s.key===e.key);if(k>=0){stopTour();selectShot(k)}}
    else if(e.key==='h'||e.key==='H'||e.key==='р'||e.key==='Р'){if(ISL.dot)unfold();else fold(nearestCorner())}
    else if(e.key==='f'||e.key==='F'||e.key==='а'||e.key==='А')toggleWalk();
    else if(e.key===' '&&document.activeElement===canvasEl){e.preventDefault();doAction()}
    if((MOVE[e.code]||e.code==='KeyQ'||e.code==='KeyE')&&!e.metaKey&&!e.ctrlKey){keys.add(e.code)}});
  window.addEventListener('keyup',e=>keys.delete(e.code));window.addEventListener('blur',()=>keys.clear());
  canvasEl.addEventListener('dblclick',()=>{if(S.shot>=0&&!DIR.on)selectShot(S.shot)});
  canvasEl.addEventListener('pointerdown',()=>{if(DIR.on)stopFilm()});
  syncUI();

  if(STILL){
    // deterministic render for exports: ?still&shot=N&t=seconds&open=seconds&tear=seconds&gust=seconds
    document.body.classList.add('clean');
    const stoneAt=params.has('stone')?Number(params.get('stone')):-1;let sto=false;
    const T=Number(params.get('t')||3),openAt=params.has('open')?Number(params.get('open')):-1;let tearAt=params.has('tear')?Number(params.get('tear')):-1;const gustAt=params.has('gust')?Number(params.get('gust')):-1;
    const dt=params.has('dt')?1/+params.get('dt'):1/60;let o=false,te=false,gu=false;const PB=new THREE.Vector3(.5,1.7,WC.z+.05);
    const advanceTo=t=>{while(S.time<t-1e-6){if(!o&&openAt>=0&&S.time>=openAt){world.openWindow();o=true}
      if(!te&&tearAt>=0&&S.time>=tearAt){world.tearFive();if(params.has('tears')&&(S.tn=(S.tn||1))<+params.get('tears')){S.tn++;tearAt+=3}else te=true}
      if(!gu&&gustAt>=0&&S.time>=gustAt){world.gust();gu=true}
      if(!sto&&stoneAt>=0&&S.time>=stoneAt){if(params.has('prep'))world.prepStory(true);world.throwStone();if(params.has('rec'))window.__rec={out:[]};sto=true}
      w.step(dt);
      if(S.pushBack){camera.position.lerp(PB,Math.min(1,dt*1.6));S.pushBack-=dt*.7;if(S.pushBack<=0)S.pushBack=0}}};
    const draw=()=>{controls.update();syncView();focusDOF();grain.uniforms.uTime.value=S.time;composer.render()};
    window.__advance=t=>{advanceTo(t);draw();return S.time};window.__advanceNoDraw=t=>{advanceTo(t);return S.time};
    window.__pt=async n=>{const c=await photoLoad();if(!c)return -1;c.start();let guard=0;while(c.samples<n&&guard++<n*40){c.step({focus:bokeh.uniforms.focus.value,time:S.time});if(c.compiling)await new Promise(r=>setTimeout(r,50))}c.step({focus:bokeh.uniforms.focus.value,time:S.time});return c.samples};
    // films, frame by frame: start a film and run it to scene time t
    window.__filmAt=(id,t,fresh)=>{if(fresh||!DIR.on||DIR.f.id!==id){const f=FILMS.find(x=>x.id===id);startFilm(f)}
      let g=0;while(DIR.on&&DIR.st<t&&g++<20000){const ws=dirStep(1/60);w.step(ws/60)}syncView();focusDOF();grain.uniforms.uTime.value=S.time;composer.render();return [DIR.st,DIR.shot]};
    window.__perf=n=>{const t0=performance.now();for(let i=0;i<n;i++)w.step(1/60);return (performance.now()-t0)/n};
    advanceTo(T);await w.wxSeek(params.has('wx')?+params.get('wx'):0,params.has('wxt')?+params.get('wxt'):1.5);w.wxApply();draw();
    window.__fly=w.flyers.filter(f=>f.active).map(f=>[f.m.position.x,f.m.position.y,f.m.position.z].map(v=>+v.toFixed(2)).concat(f.rest?1:0));
    document.getElementById('loader').classList.add('done');document.body.dataset.ready='1';
    return;
  }

  const clock=new THREE.Clock();
  function frame(){
    if(window.__hold){clock.getDelta();requestAnimationFrame(frame);return}
    const dt=Math.min(.05,clock.getDelta()),pc=PHOTO.ctl,frozen=!!(pc&&pc.active);
    const ws=DIR.on?dirStep(dt):1;
    if(!frozen&&ws>0)world.step((reduce&&!DIR.on?dt*.5:dt)*ws);
    if(ENTRY){ENTRY.t+=dt;const k=Math.min(1,ENTRY.t/ENTRY.d),e=1-Math.pow(1-k,3);camera.position.lerpVectors(ENTRY.P0,ENTRY.P,e);camera.fov=ENTRY.f0+(ENTRY.f-ENTRY.f0)*e;camera.updateProjectionMatrix();
      canvasEl.style.filter=k<1?`blur(${((1-e)*(1-e)*12).toFixed(2)}px)`:'';if(k>=1){ENTRY=null;canvasEl.style.filter=''}}
    if(flying&&!DIR.on){flying.t=Math.min(1,flying.t+dt/1.3);const e=flying.t<.5?4*flying.t**3:1-Math.pow(-2*flying.t+2,3)/2;
      camera.position.lerpVectors(flying.p0,flying.P,e);controls.target.lerpVectors(flying.t0,flying.T,e);camera.fov=flying.f0+(flying.F-flying.f0)*e;camera.updateProjectionMatrix();
      if(flying.t>=1)flying=null}
    if(S.pushBack&&!DIR.on){camera.position.lerp(new THREE.Vector3(.5,1.7,WC.z+.05),Math.min(1,dt*1.6));S.pushBack-=dt*.7;if(S.pushBack<=0)S.pushBack=0}
    if(!DIR.on){if(WALK.on)walkStep(dt);else walk(dt)}if(S.storyT>=0&&S.storyT<12&&Math.floor(S.storyT*2)!==S._lastLbl){S._lastLbl=Math.floor(S.storyT*2);refreshAction()}
    if(!WALK.on&&!DIR.on)controls.update();
    // rest detection: the camera has not moved, nothing is flying, the stone is not in the air
    {const now=performance.now(),p=camera.position,q=camera.quaternion,key=[p.x,p.y,p.z,q.x,q.y,q.z,q.w,camera.fov].map(v=>v.toFixed(4)).join();
      const busy=DIR.on||!!ENTRY||!!flying||S.shake>.002||!!S.pushBack||(S.storyT>=0&&S.storyT<14)||(WALK.on&&(WALK.vel.lengthSq()>1e-6||keys.size>0||WALK.joy.x||WALK.joy.y))||document.hidden;
      if(key!==PHOTO.key||busy){PHOTO.key=key;PHOTO.last=now;if(pc&&pc.active)pc.stop()}
      if(PHOTO.on&&!busy&&now-PHOTO.last>750){if(!pc)photoLoad();else if(!pc.active)pc.start()}
      // a few seconds after opening, build the traced scene once so its shader is compiled before it is needed
      else if(PHOTO.on&&!PHOTO.warmed&&now>PHOTO.t0+4000){PHOTO.warmed=true;photoLoad().then(c=>{if(c&&!c.active)c.prewarm()})}}
    const pc2=PHOTO.ctl;
    if(pc2&&pc2.active){if(!pc2.shown){syncView();focusDOF();grain.uniforms.uTime.value=S.time;composer.render();if(pc2.ready)photoFade()}focusDOF();pc2.step({focus:bokeh.uniforms.focus.value,time:S.time})}
    else{syncView();focusDOF();grain.uniforms.uTime.value=S.time;composer.render()}
    requestAnimationFrame(frame);
  }
  frame();
  loadmsg.textContent='Вешаю погоду за окном…';
  for(let k=0;k<300&&!world.wxReady();k++)await new Promise(r=>setTimeout(r,50));LOAD.vid=1;
  loadmsg.textContent='Настраиваю свет…';LOAD.item='шейдеры и тени';
  try{if(renderer.compileAsync)await Promise.race([renderer.compileAsync(scene,view),new Promise(r=>setTimeout(r,6000))])}catch(e){}
  LOAD.cmp=1;loadmsg.textContent='Открываю дверь…';LOAD.item='';LOAD.fin=true;
  await new Promise(r=>{const w=()=>LOAD.shown>=.995?r():requestAnimationFrame(w);w()});
  // the way in: from a step behind, out of a blur, the camera moves forward to its place
  { const P=camera.position.clone(),T=controls.target.clone(),d=new THREE.Vector3().subVectors(T,P).normalize(),P0=P.clone().addScaledVector(d,-1.1);P0.y-=.06;
    P0.x=THREE.MathUtils.clamp(P0.x,lim.x[0],lim.x[1]);P0.z=THREE.MathUtils.clamp(P0.z,lim.z[0],lim.z[1]);
    if(!reduce){ENTRY={t:0,d:2.4,P0,P,f0:camera.fov+9,f:camera.fov};camera.position.copy(P0)} }
  document.getElementById('loader').classList.add('done');
}).catch(e=>{console.error(e);loadmsg.className='err';loadmsg.textContent='Сцена не собралась: '+e.message});
