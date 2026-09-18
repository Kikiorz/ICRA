import * as THREE from 'three';
import {OrbitControls} from '../vendor/three/OrbitControls.js';

const $ = id => document.getElementById(id);
const run = new URLSearchParams(location.search).get('run') || 'results/vggt_cup_single';
const fail = e => { $('error').style.display='block'; $('error').textContent=String(e); $('status').textContent='加载失败'; };
async function fetchOK(url) { const r=await fetch(url); if(!r.ok) throw new Error(`${url}: HTTP ${r.status}`); return r; }
const linear = c => c <= .04045 ? c/12.92 : ((c+.055)/1.055)**2.4;

async function main() {
  if(!/^results\/[A-Za-z0-9_-]+$/.test(run)) throw new Error('无效的结果目录');
  $('inputImage').src=`${run}/model_input.png`; $('depthImage').src=`${run}/depth_preview.png`;
  $('plyLink').href=`${run}/pointcloud.ply`; $('rawLink').href=`${run}/predictions.npz`; $('cameraLink').href=`${run}/camera.json`;
  const [meta,cam,buffer] = await Promise.all([
    fetchOK(`${run}/metadata.json`).then(r=>r.json()), fetchOK(`${run}/camera.json`).then(r=>r.json()),
    fetchOK(`${run}/pointcloud.ply`).then(r=>r.arrayBuffer())
  ]);
  const head=new TextDecoder().decode(new Uint8Array(buffer,0,Math.min(4096,buffer.byteLength)));
  const end=head.indexOf('end_header\n');
  if(end<0 || !head.includes('format binary_little_endian 1.0')) throw new Error('不支持的 PLY 格式');
  const count=Number(head.match(/element vertex (\d+)/)?.[1]), offset=end+11, stride=23;
  if(!count || offset+count*stride!==buffer.byteLength) throw new Error('PLY 长度与顶点数不一致');
  const positions=new Float32Array(count*3), colors=new Float32Array(count*3), conf=new Float32Array(count), uv=new Uint16Array(count*2);
  const dv=new DataView(buffer), E=cam.extrinsic;
  for(let i=0;i<count;i++) {
    const p=offset+i*stride, x=dv.getFloat32(p,true),y=dv.getFloat32(p+4,true),z=dv.getFloat32(p+8,true);
    // Raw PLY stays in VGGT world coordinates. Viewer transforms to the input
    // camera, then OpenCV -> Three.js: x unchanged, y/z negated.
    positions[i*3]=E[0][0]*x+E[0][1]*y+E[0][2]*z+E[0][3];
    positions[i*3+1]=-(E[1][0]*x+E[1][1]*y+E[1][2]*z+E[1][3]);
    positions[i*3+2]=-(E[2][0]*x+E[2][1]*y+E[2][2]*z+E[2][3]);
    for(let j=0;j<3;j++) colors[i*3+j]=linear(dv.getUint8(p+12+j)/255);
    conf[i]=dv.getFloat32(p+15,true);uv[i*2]=dv.getUint16(p+19,true);uv[i*2+1]=dv.getUint16(p+21,true);
  }
  const viewport=$('viewport'),scene=new THREE.Scene(); scene.background=new THREE.Color('#0c131c');
  const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2));
  viewport.appendChild(renderer.domElement);
  const fov=2*Math.atan(cam.image_size[1]/(2*cam.intrinsic[1][1]))*180/Math.PI;
  const camera=new THREE.PerspectiveCamera(fov,1,.001,10000);
  const controls=new OrbitControls(camera,renderer.domElement);controls.enableDamping=true;
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(positions,3));geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));
  const material=new THREE.PointsMaterial({size:2,vertexColors:true,sizeAttenuation:false});
  scene.add(new THREE.Points(geometry,material));
  let roi=null,shown=[],center=new THREE.Vector3(),radius=1;
  function update() {
    shown=[];const threshold=meta.confidence_percentiles[Number($('confidence').value)];
    for(let i=0;i<count;i++) {
      if(conf[i]<threshold) continue;
      if(roi && (uv[i*2]<roi[0] || uv[i*2]>roi[2] || uv[i*2+1]<roi[1] || uv[i*2+1]>roi[3])) continue;
      shown.push(i);
    }
    geometry.setIndex(shown);
    $('confidenceValue').textContent=`${$('confidence').value}%`;
    $('status').textContent=`${shown.length.toLocaleString()} / ${count.toLocaleString()} 个点`;
    document.body.dataset.points=String(shown.length);
  }
  function bounds() {
    if(!shown.length) return;
    const box=new THREE.Box3();const p=new THREE.Vector3();
    for(const i of shown) box.expandByPoint(p.fromArray(positions,i*3));
    box.getCenter(center);radius=Math.max(box.getSize(p).length()/2,.0001);
    camera.near=Math.max(radius/10000,.000001);camera.far=radius*1000;camera.updateProjectionMatrix();
  }
  function view(which) {
    bounds(); controls.target.copy(center);camera.up.set(0,1,0);
    const distance=radius/Math.sin(THREE.MathUtils.degToRad(camera.fov)/2)*1.1;
    if(which==='front') {camera.position.set(0,0,0);controls.target.set(0,0,center.z);}
    else {
      const direction=which==='side'?new THREE.Vector3(1,0,0):which==='back'?new THREE.Vector3(0,0,-1):new THREE.Vector3(.55,.20,1).normalize();
      camera.position.copy(center).addScaledVector(direction,distance);
    }
    camera.lookAt(controls.target);controls.update();
    document.body.dataset.view=which;
  }
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>view(b.dataset.view));
  $('fit').onclick=()=>view('oblique');
  $('confidence').oninput=update;
  $('pointSize').oninput=()=>{material.size=Number($('pointSize').value);$('sizeValue').textContent=$('pointSize').value;};
  const overlay=$('roiCanvas'),ctx=overlay.getContext('2d');overlay.width=cam.image_size[0];overlay.height=cam.image_size[1];
  const getPixel=e=>{const r=overlay.getBoundingClientRect();return [Math.max(0,Math.min(overlay.width,(e.clientX-r.left)/r.width*overlay.width)),Math.max(0,Math.min(overlay.height,(e.clientY-r.top)/r.height*overlay.height))];};
  let start=null;
  overlay.onpointerdown=e=>{start=getPixel(e);overlay.setPointerCapture(e.pointerId);};
  overlay.onpointermove=e=>{if(!start)return;const p=getPixel(e);ctx.clearRect(0,0,overlay.width,overlay.height);ctx.strokeStyle='#61ffe0';ctx.lineWidth=3;ctx.strokeRect(start[0],start[1],p[0]-start[0],p[1]-start[1]);};
  overlay.onpointerup=e=>{if(!start)return;const p=getPixel(e);roi=[Math.min(start[0],p[0]),Math.min(start[1],p[1]),Math.max(start[0],p[0]),Math.max(start[1],p[1])];start=null;if(roi[2]-roi[0]<3||roi[3]-roi[1]<3)roi=null;update();view('oblique');};
  $('clearRoi').onclick=()=>{roi=null;ctx.clearRect(0,0,overlay.width,overlay.height);update();view('oblique');};
  function resize(){const w=viewport.clientWidth,h=viewport.clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}
  new ResizeObserver(resize).observe(viewport);resize();update();view(new URLSearchParams(location.search).get('view')||'oblique');
  renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
  document.body.dataset.ready='true';document.title='VGGT 点云 · 已加载';
}
main().catch(fail);
