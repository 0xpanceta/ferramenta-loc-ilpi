"""Telemetria do app — registra cada sessão e ação com origem.

Todo evento vira uma linha JSON no stdout (visível no console "Manage app" do
Streamlit Cloud). Se existir o segredo `telemetria.webhook_url`, o evento também
é enviado por POST em segundo plano (fire-and-forget) para registro permanente —
ex.: automação "When webhook received" do Airtable.

Nenhum dado de formulário é coletado; registra-se origem técnica (IP, navegador,
idioma, fuso) e interações (navegação, filtros, buscas, downloads).
"""
import json
import threading
import urllib.request
from datetime import datetime, timezone

import streamlit as st


def _webhook_url():
    try:
        return st.secrets["telemetria"]["webhook_url"]
    except Exception:
        return None


def _sessao_id() -> str:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        return ctx.session_id[:12] if ctx else "desconhecida"
    except Exception:
        return "desconhecida"


def _origem() -> dict:
    """Origem técnica da sessão: quem (anonimamente), de onde e com o quê."""
    try:
        headers = dict(st.context.headers or {})
    except Exception:
        headers = {}
    pega = lambda *ks: next((headers[k] for k in ks if k in headers), None)

    def _privado(v):
        import ipaddress
        try:
            return not ipaddress.ip_address(v.replace("::ffff:", "")).is_global
        except ValueError:
            return True

    try:
        ip = st.context.ip_address
    except Exception:
        ip = None
    xff = pega("X-Forwarded-For", "x-forwarded-for") or ""
    if not ip or _privado(str(ip)):
        # atrás do proxy do Streamlit Cloud, ip_address e o início do
        # X-Forwarded-For são IPs internos — pega o primeiro IP público da cadeia
        candidatos = [c.strip() for c in xff.split(",") if c.strip()]
        publicos = [c for c in candidatos if not _privado(c)]
        ip = publicos[0] if publicos else (candidatos[0] if candidatos else ip)
    # o Streamlit Community Cloud não repassa o IP do visitante (verificado:
    # só chegam IPs internos 10.x) — deixa vazio em vez de registrar IP de proxy
    if ip and _privado(str(ip)):
        ip = None
    ua = pega("User-Agent", "user-agent") or ""
    robo = any(t in ua.lower() for t in ("headless", "prerender", "bot", "crawler", "spider"))
    return {
        "ip": ip,
        "robo": robo or None,
        "user_agent": pega("User-Agent", "user-agent"),
        "referer": pega("Referer", "referer"),
        "idioma": getattr(st.context, "locale", None),
        "fuso": getattr(st.context, "timezone", None),
        "url": str(getattr(st.context, "url", "") or ""),
        "embutido": bool(getattr(st.context, "is_embedded", False)),
    }


def _enviar_webhook(url: str, payload: dict):
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass  # telemetria nunca derruba o app


def registrar(evento: str, **dados):
    """Registra um evento: linha JSON no stdout + webhook (se configurado)."""
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sessao": _sessao_id(),
        "evento": evento,
        "dados": {k: v for k, v in dados.items() if v not in (None, "")},
    }
    try:
        origem = st.session_state.get("_tele_origem") or {}
        payload["ip"] = origem.get("ip")
    except Exception:
        pass
    print("TELEMETRIA " + json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    url = _webhook_url()
    if url:
        origem = st.session_state.get("_tele_origem") or {}
        plano = {  # payload plano: mapeia 1:1 com os campos da tabela de logs
            "ts": payload["ts"], "sessao": payload["sessao"], "evento": evento,
            "dados": json.dumps(payload["dados"], ensure_ascii=False, default=str),
            "ip": str(origem.get("ip") or ""),
            "user_agent": str(origem.get("user_agent") or ""),
            "idioma": str(origem.get("idioma") or ""),
            "fuso": str(origem.get("fuso") or ""),
            "referer": str(origem.get("referer") or ""),
            "url": str(origem.get("url") or ""),
        }
        threading.Thread(target=_enviar_webhook, args=(url, plano), daemon=True).start()


def registrar_sessao():
    """Uma vez por sessão: evento de abertura com a origem completa."""
    if st.session_state.get("_tele_sessao_ok"):
        return
    origem = _origem()
    st.session_state["_tele_origem"] = origem
    st.session_state["_tele_sessao_ok"] = True
    registrar("sessao_inicio", **origem)


def registrar_delta(chave: str, valor, evento: str = "interacao"):
    """Registra só quando o valor muda (o Streamlit reexecuta o script inteiro
    a cada interação — sem o delta, cada clique geraria dezenas de eventos)."""
    prev = st.session_state.setdefault("_tele_prev", {})
    serial = json.dumps(valor, ensure_ascii=False, sort_keys=True, default=str)
    if prev.get(chave) == serial:
        return
    primeira_vez = chave not in prev
    prev[chave] = serial
    if primeira_vez:
        return  # primeiro render = estado default, não é ação do usuário
    registrar(evento, campo=chave, valor=valor)
