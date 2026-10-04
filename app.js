const form = document.getElementById('predictionForm');
const emptyState = document.getElementById('emptyState');
const resultContent = document.getElementById('resultContent');
const loading = document.getElementById('loading');
const cropPhoto = document.getElementById('cropPhoto');
const photoPreviewWrap = document.getElementById('photoPreviewWrap');
const photoPreview = document.getElementById('photoPreview');
const removePhoto = document.getElementById('removePhoto');

if (cropPhoto) cropPhoto.addEventListener('change', () => {
  const file = cropPhoto.files?.[0];
  if (!file) return;
  if (!file.type.startsWith('image/')) { alert('Please choose an image file.'); cropPhoto.value=''; return; }
  const reader = new FileReader();
  reader.onload = e => { photoPreview.src = e.target.result; photoPreviewWrap.classList.remove('hidden'); };
  reader.readAsDataURL(file);
});
if (removePhoto) removePhoto.addEventListener('click', () => { cropPhoto.value=''; photoPreview.src=''; photoPreviewWrap.classList.add('hidden'); });

form.addEventListener('submit', async (e) => {
  e.preventDefault(); loading.classList.remove('hidden');
  try {
    const response = await fetch('/predict', {method:'POST', body:new FormData(form)});
    const result = await response.json();
    if (!result.success) throw new Error(result.message || 'Prediction failed');
    emptyState.classList.add('hidden'); resultContent.classList.remove('hidden');
    document.getElementById('resultPanel').scrollIntoView({behavior:'smooth', block:'start'});
    document.getElementById('resultCrop').textContent = result.crop;
    document.getElementById('growthStage').textContent = result.growth_stage;
    document.getElementById('likelyPest').textContent = result.likely_pest;
    document.getElementById('confidence').textContent = result.confidence + '%';
    document.getElementById('confidenceText').textContent = result.confidence + '%';
    const badge = document.getElementById('riskBadge'); badge.textContent=result.risk.toUpperCase(); badge.className='risk '+result.risk.toLowerCase();
    document.getElementById('confidenceBar').style.width=result.confidence+'%';
    const list=document.getElementById('adviceList'); list.innerHTML='';
    result.advice.forEach(x=>{const li=document.createElement('li');li.textContent=x;list.appendChild(li);});
    const ir=result.image_result;
    if (ir?.available) {
      document.getElementById('imageDisease').textContent = `${ir.crop} — ${ir.disease}`;
      document.getElementById('imageConfidence').textContent = ir.confidence+'%';
      document.getElementById('imagePrevention').textContent = ir.prevention;
    } else {
      document.getElementById('imageDisease').textContent='Image model not trained';
      document.getElementById('imageConfidence').textContent='—';
      document.getElementById('imagePrevention').textContent=ir?.message || 'Run download_real_model.py first.';
    }
    document.getElementById('preventionText').textContent = ir?.available ? ir.prevention : result.advice[3];
    document.getElementById('fusionText').textContent = result.fusion || '—';
  } catch(err) { alert('Prediction failed: '+err.message); console.error(err); }
  finally { loading.classList.add('hidden'); }
});

async function getLiveWeather(lat, lon) {
  const url=`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m&timezone=auto`;
  const r=await fetch(url); if(!r.ok) throw new Error('Weather service unavailable');
  return r.json();
}
function setWeatherUI(lat,lon,data){
  const c=data.current; const temp=c.temperature_2m, hum=c.relative_humidity_2m, rain=c.rain ?? c.precipitation ?? 0, wind=c.wind_speed_10m;
  document.getElementById('locationStatus').textContent=`Detected: ${lat.toFixed(4)}, ${lon.toFixed(4)}`;
  document.getElementById('locationTitle').textContent='Live field location detected';
  document.getElementById('coords').textContent=`Latitude ${lat.toFixed(5)} • Longitude ${lon.toFixed(5)} • ${data.timezone || 'local time'}`;
  document.getElementById('weatherCards').classList.remove('hidden');
  document.getElementById('liveTemp').textContent=temp+' °C'; document.getElementById('liveHumidity').textContent=hum+' %'; document.getElementById('liveRain').textContent=rain+' mm'; document.getElementById('liveWind').textContent=wind+' km/h';
  document.getElementById('bigTemp').textContent=temp+' °C'; document.getElementById('bigHumidity').textContent=hum+' %'; document.getElementById('bigRain').textContent=rain+' mm'; document.getElementById('bigWind').textContent=wind+' km/h';
  document.getElementById('temperature').value=temp; document.getElementById('humidity').value=hum; document.getElementById('rainfall').value=rain;
}
document.getElementById('locationBtn').addEventListener('click',()=>{
  const status=document.getElementById('locationStatus'); status.textContent='Requesting browser location permission…';
  if(!navigator.geolocation){status.textContent='Geolocation is not supported by this browser.';return;}
  navigator.geolocation.getCurrentPosition(async pos=>{try{const data=await getLiveWeather(pos.coords.latitude,pos.coords.longitude);setWeatherUI(pos.coords.latitude,pos.coords.longitude,data);}catch(e){status.textContent='Location found, but weather could not be loaded.';}}, err=>{status.textContent='Location permission was denied or unavailable. You can enter values manually.';},{enableHighAccuracy:true,timeout:10000,maximumAge:300000});
});
