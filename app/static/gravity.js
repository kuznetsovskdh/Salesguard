(function(){
  const TOKENS = [
    '2024-03-15','#8472','SKU-441','1 299 ₽','C-0091','TXN-4421',
    '2023-11-02','#3301','SKU-887','549 ₽','C-4872','revenue',
    'sale_date','product','qty: 42','customer_id','2024-01-28',
    '12 400 ₽','#1147','region: EU','TXN-7741','SKU-220','NaN',
    '2023-08-14','#9934','340 ₽','C-0018','SKU-115','null',
    '4 750 ₽','2024-06-01','#4421','SKU-998','ID-8801','280 ₽',
    'avg_check','2023-12-25','#7753','99 ₽','TXN-5591','SKU-674',
    '1 870 ₽','ID-3310','2024-04-19','#2208','qty: 7','620 ₽',
  ];
  const GARBAGE = new Set(['NaN','null','???','—','#ERR','undefined']);

  const canvas = document.getElementById('gravity-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let W, H, particles = [], phase = 'fall', phaseTimer = 0;
  const PHASE_HOLD = 140, PHASE_DRAIN = 50;

  function resize(){ W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
  resize();
  window.addEventListener('resize', resize);

  function spawn(){
    const text = TOKENS[Math.floor(Math.random()*TOKENS.length)];
    const garbage = GARBAGE.has(text);
    return {
      x: Math.random()*W, y: -24,
      vy: 0.7 + Math.random()*2.0,
      vx: (Math.random()-.5)*0.35,
      text, garbage,
      alpha: 0.45 + Math.random()*0.35,
      resting: false, restY: 0,
      draining: false, drainVy: 0,
    };
  }

  const FLOOR = ()=> H - 56;
  let frame = 0;

  function update(){
    frame++;
    if(phase==='fall' && frame%5===0 && particles.length<72) particles.push(spawn());

    if(phase==='fall'){
      const resting = particles.filter(p=>p.resting).length;
      if(particles.length>38 && resting/particles.length>0.72){ phase='hold'; phaseTimer=PHASE_HOLD; }
    } else if(phase==='hold'){
      if(--phaseTimer<=0){
        phase='drain'; phaseTimer=PHASE_DRAIN;
        particles.forEach(p=>{ p.draining=true; p.drainVy=1.5+Math.random()*3.5; });
      }
    } else if(phase==='drain'){
      phaseTimer--;
      if(phaseTimer<=0 && particles.every(p=>p.y>H+50)){ particles=[]; phase='fall'; frame=0; }
    }

    particles.forEach(p=>{
      if(p.draining){ p.drainVy+=0.55; p.y+=p.drainVy; p.alpha=Math.max(0,p.alpha-0.018); return; }
      if(p.resting) return;
      p.vy+=0.16; p.y+=p.vy; p.x+=p.vx;
      p.x=Math.max(4,Math.min(W-4,p.x));
      let stackY=FLOOR();
      particles.forEach(o=>{
        if(o===p||!o.resting) return;
        if(Math.abs(o.x-p.x)<100) stackY=Math.min(stackY, o.restY-16);
      });
      if(p.y>=stackY){ p.y=stackY; p.vy=0; p.vx=0; p.resting=true; p.restY=stackY; }
    });
  }

  function draw(){
    ctx.clearRect(0,0,W,H);
    ctx.font='11px "IBM Plex Mono",monospace';
    particles.forEach(p=>{
      ctx.globalAlpha=p.alpha;
      ctx.fillStyle=p.garbage?'#ff4060':p.resting?'#00e87a':'#6b6b8a';
      ctx.fillText(p.text, p.x, p.y);
    });
    ctx.globalAlpha=1;
  }

  function loop(){ update(); draw(); requestAnimationFrame(loop); }
  loop();
})();

// Переиспользуемый DVD-bounce для .dvd-bar - используется и во время
// "uploading to server" (index.html), и как фон рейтинг-модалки (result.html).
// Визуально это прямой луч (длинная тонкая полоса), а не квадратный "лого" -
// от DVD-анимации здесь только траектория (отражение от 4 стен экрана).
// Столкновения считаются по осевому bounding-box'у (без вращения), а сам
// элемент довдоворачивается transform:rotate() по направлению вектора
// скорости, чтобы луч всегда "смотрел" туда, куда летит.
window.startDvdBounce = function(el, opts){
  opts = opts || {};
  const speed = opts.speed || 3.9;
  const w = el.offsetWidth, h = el.offsetHeight;
  let x = Math.random()*(window.innerWidth-w), y = Math.random()*(window.innerHeight-h);
  let vx = (Math.random()<0.5?-1:1)*speed, vy = (Math.random()<0.5?-1:1)*speed*0.55;
  let running = true;
  (function step(){
    if(!running) return;
    x+=vx; y+=vy;
    if(x<=0||x+w>=window.innerWidth) vx=-vx;
    if(y<=0||y+h>=window.innerHeight) vy=-vy;
    x=Math.max(0,Math.min(window.innerWidth-w,x));
    y=Math.max(0,Math.min(window.innerHeight-h,y));
    const angle=Math.atan2(vy,vx)*180/Math.PI;
    el.style.transform=`translate(${x}px,${y}px) rotate(${angle}deg)`;
    requestAnimationFrame(step);
  })();
  return function stop(){ running=false; };
};
