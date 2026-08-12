"""
Cliente para la API de H2H GG League (hudstats), descubierta inspeccionando
las peticiones de red reales que hace https://h2hggl.com

IMPORTANTE - USO RESPONSABLE:
- Esta es una API interna no documentada públicamente, pensada para alimentar
  el propio sitio web. No está pensada para scraping masivo.
- Usa siempre pausas entre peticiones (time.sleep) para no sobrecargar su
  servidor ni arriesgarte a que te bloqueen la IP.
- Revisa periódicamente si la estructura de la respuesta cambia (pueden
  actualizar su frontend y con ello estos endpoints en cualquier momento).
- No hay garantía de que estos endpoints sigan disponibles ni de su fiabilidad;
  trátalos como lo que son: ingeniería inversa de un frontend ajeno.
"""
import requests
import time

BASE_URL = "https://api-h2h.hudstats.com/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}
REQUEST_DELAY = 1.5   # segundos entre peticiones - SÉ RESPETUOSO con su servidor
MAX_RETRIES = 4        # reintentos ante 429 (Too Many Requests)
BACKOFF_BASE_SECONDS = 20  # espera tras un 429: 20s, 40s, 80s, 160s...


def _get(path: str, params: dict | None = None):
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(f"{BASE_URL}{path}", headers=HEADERS, params=params, timeout=15)
        if resp.status_code == 429:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()  # se acabaron los reintentos, propaga el error
            wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(f"    (429 recibido, esperando {wait}s antes de reintentar...)")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()


def get_schedule_for_date(date_str: str):
    """
    date_str: 'YYYY-MM-DD'. Devuelve TODOS los partidos de ese día
    (pasados = con resultado, futuros = programados).
    """
    return _get("/schedule/nba", params={"date": f"{date_str}T00:00:00+02:00"})


def get_live_matches():
    return _get("/live/nba")


def get_all_participant_names():
    return _get("/participant/nba/names")


def get_participant_stats(participant: str):
    """Estadísticas de carrera ACUMULADAS a fecha de hoy. OJO: no point-in-time,
    no usar directamente como feature de entrenamiento histórico (data leakage).
    Útil solo como snapshot de 'nivel actual' para predecir partidos futuros."""
    return _get("/participant/nba/stats", params={"participant": participant})


def get_match_stats(external_id: str):
    """Estadísticas por cuarto de un partido concreto (pace, tiros, etc.)."""
    return _get("/match/stats/nba", params={"external_id": external_id})
