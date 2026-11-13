"""Análise de Localização para ILPIs — Brasil.

Ferramenta de consultoria para avaliar o potencial de mercado de Instituições
de Longa Permanência para Idosos (ILPIs), com base no Censo IBGE 2022:
população 70+, renda do responsável pelo domicílio e estimativa de idosos
em faixas de renda alta, por estado → município → distrito → bairro.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Identidade visual
# ---------------------------------------------------------------------------
COR_GRAFICO = "#0D9488"   # marca de dados (validada p/ contraste e CVD)
COR_TEXTO_SEC = "#5B6B68"
LIMITE_RENDA = 8000

st.set_page_config(
    page_title="Análise de Localização · ILPIs no Brasil",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; }
    h1 { letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: #FBFDFC;
        border: 1px solid #E3EAE8;
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
    }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        padding-top: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Formatação de números no padrão brasileiro
# ---------------------------------------------------------------------------
def fmt_int(v) -> str:
    return f"{v:,.0f}".replace(",", ".")


def fmt_brl(v) -> str:
    s = f"{v:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return f"R$ {s}"


# ---------------------------------------------------------------------------
# Carregamento dos dados (Censo IBGE 2022)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def carregar_csv(caminho, sep=";"):
    """Lê CSV tentando múltiplos encodings (arquivos do IBGE variam)."""
    for encoding in ["utf-8", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            df = pd.read_csv(caminho, sep=sep, encoding=encoding, dtype=str)
            df.columns = df.columns.str.strip('"')
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].str.strip('"')
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível decodificar {caminho}")


@st.cache_data(show_spinner=False)
def preparar_municipios():
    """Base municipal: população 70+ (V01041 agregado por distrito) + renda."""
    dist = carregar_csv("Dados IBGE/Distrito/Agregados_por_distritos_demografia_BR.csv")
    dist["CD_MUN_extracted"] = dist["CD_DIST"].str[:7]
    dist["V01041_num"] = pd.to_numeric(dist["V01041"], errors="coerce").fillna(0)
    mun_idosos = dist.groupby("CD_MUN_extracted")["V01041_num"].sum().reset_index()
    mun_idosos.columns = ["CD_MUN", "elderly_total"]

    renda = carregar_csv("Dados IBGE/Municipio/renda media municipio.csv")
    info = carregar_csv("Dados IBGE/Municipio/infos sobre municipios.csv")
    demo = carregar_csv("Dados IBGE/Municipio/demografia municipios.csv")

    demo["total_population"] = pd.to_numeric(demo.get("V00005", 0), errors="coerce").fillna(0)

    cols_renda = ["CD_MUN"]
    if "Rendaa media do responsavel" in renda.columns:
        renda["avg_income_raw"] = pd.to_numeric(
            renda["Rendaa media do responsavel"].astype(str).str.replace(",", "."),
            errors="coerce",
        )
        cols_renda.append("avg_income_raw")
    elif "V06004" in renda.columns:
        renda["avg_income_raw"] = pd.to_numeric(renda["V06004"], errors="coerce")
        cols_renda.append("avg_income_raw")

    if "Variancia" in renda.columns:
        renda["income_variance_raw"] = pd.to_numeric(
            renda["Variancia"].astype(str).str.replace(",", "."), errors="coerce"
        )
        cols_renda.append("income_variance_raw")
    elif "V06005" in renda.columns:
        renda["income_variance_raw"] = pd.to_numeric(renda["V06005"], errors="coerce")
        cols_renda.append("income_variance_raw")

    base = (
        mun_idosos
        .merge(info[["CD_MUN", "NM_MUN", "NM_UF"]], on="CD_MUN", how="left")
        .merge(demo[["CD_MUN", "total_population"]], on="CD_MUN", how="left")
        .merge(renda[cols_renda], on="CD_MUN", how="left")
    )
    base["avg_income"] = base.get("avg_income_raw", 0)
    base["income_variance"] = base.get("income_variance_raw", 0)
    base["income_std"] = np.sqrt(base["income_variance"].fillna(0))
    base["name"] = base["NM_MUN"]
    base["state"] = base["NM_UF"]
    base["elderly_percentage"] = np.where(
        base["total_population"] > 0,
        (base["elderly_total"] / base["total_population"] * 100).round(1),
        0,
    )
    return base


@st.cache_data(show_spinner=False)
def preparar_hierarquia():
    """Bases de distritos e bairros com população 70+ e renda."""
    dist = carregar_csv("Dados IBGE/Distrito/Agregados_por_distritos_demografia_BR.csv")
    dist_renda = carregar_csv("Dados IBGE/Distrito/Agregados_por_distritos_renda_responsavel_BR.csv")
    bairro = carregar_csv("Dados IBGE/Bairro/Agregados_por_bairros_demografia_BR.csv")
    bairro_renda = carregar_csv("Dados IBGE/Bairro/Agregados_por_bairros_renda_responsavel_BR.csv")
    hierarquia = carregar_csv("Dados IBGE/Bairro/Info sobre os bairros.csv")

    dist = dist.copy()
    dist["elderly_total"] = pd.to_numeric(dist["V01041"], errors="coerce").fillna(0)
    dist["CD_MUN"] = dist["CD_DIST"].str[:7]
    dist = dist.merge(dist_renda[["CD_DIST", "V06004", "V06005"]], on="CD_DIST", how="left")
    dist["avg_income"] = pd.to_numeric(
        dist["V06004"].astype(str).str.replace(",", "."), errors="coerce"
    ).fillna(0)
    dist["income_variance"] = pd.to_numeric(
        dist["V06005"].astype(str).str.replace(",", "."), errors="coerce"
    ).fillna(0)
    dist["income_std"] = np.sqrt(dist["income_variance"])

    bairro = bairro.copy()
    bairro["elderly_total"] = pd.to_numeric(bairro["V01041"], errors="coerce").fillna(0)
    bairro = bairro.merge(bairro_renda[["CD_BAIRRO", "V06004", "V06005"]], on="CD_BAIRRO", how="left")
    bairro["avg_income"] = pd.to_numeric(
        bairro["V06004"].astype(str).str.replace(",", "."), errors="coerce"
    ).fillna(0)
    bairro["income_variance"] = pd.to_numeric(
        bairro["V06005"].astype(str).str.replace(",", "."), errors="coerce"
    ).fillna(0)
    bairro["income_std"] = np.sqrt(bairro["income_variance"])

    cols_hier = ["CD_BAIRRO", "CD_DIST", "NM_DIST", "CD_MUN", "NM_MUN", "NM_UF"]
    cols_disponiveis = [c for c in cols_hier if c in hierarquia.columns]
    if cols_disponiveis:
        bairro = bairro.merge(hierarquia[cols_disponiveis], on="CD_BAIRRO", how="left")

    return dist, bairro


# ---------------------------------------------------------------------------
# Estimativa de idosos com renda acima de R$ 8 mil (distribuição log-normal)
# ---------------------------------------------------------------------------
def _erf(x):
    """Aproximação de Abramowitz & Stegun, vetorizada (evita depender de scipy)."""
    sinal = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)
    return sinal * y


def estimar_idosos_acima(pop, media, desvio, limite=LIMITE_RENDA):
    """Modela a renda como log-normal e estima quantos idosos 70+ do local
    vivem em domicílios cujo responsável ganha acima do limite.

    O CV é limitado a 1,0 porque a variância do IBGE frequentemente produz
    valores irreais (>200%)."""
    pop = pd.to_numeric(pop, errors="coerce").fillna(0).to_numpy(dtype=float)
    media = pd.to_numeric(media, errors="coerce").fillna(0).to_numpy(dtype=float)
    desvio = pd.to_numeric(desvio, errors="coerce").fillna(0).to_numpy(dtype=float)

    valido = (pop > 0) & (media > 0) & (desvio > 0)
    media_s = np.where(valido, media, 1.0)
    cv = np.minimum(np.where(valido, desvio / media_s, 0.0), 1.0)
    sigma = np.sqrt(np.log1p(cv**2))
    sigma_s = np.where(sigma > 0, sigma, 1.0)
    mu = np.log(media_s) - sigma**2 / 2
    z = (np.log(limite) - mu) / sigma_s
    prob_abaixo = 0.5 * (1 + _erf(z / np.sqrt(2)))
    resultado = np.round((1 - prob_abaixo) * pop)
    return np.where(valido, resultado, 0.0)


@st.cache_data(show_spinner=False)
def montar_bases():
    """Bases finais com a estimativa de mercado já calculada."""
    municipios = preparar_municipios()
    distritos, bairros = preparar_hierarquia()
    for df in (municipios, distritos, bairros):
        df["elders_above_8k"] = estimar_idosos_acima(
            df["elderly_total"], df["avg_income"], df["income_std"]
        )
    return municipios, distritos, bairros


def coluna_nome_bairro(df):
    for c in ("NM_BAIRRO", "NM_BAIRRO_x", "NM_BAIRRO_y"):
        if c in df.columns:
            return c
    return df.columns[0]


# ---------------------------------------------------------------------------
# Componentes de interface
# ---------------------------------------------------------------------------
def _reset_keys(*chaves):
    def cb():
        for k in chaves:
            st.session_state[k] = None
    return cb


def _limpar_tudo():
    for k in ("nav_uf", "nav_mun", "nav_dist", "nav_bairro"):
        st.session_state[k] = None
    st.session_state["f_pop"] = 0
    st.session_state["f_renda"] = 0.0


def select_com_placeholder(rotulo, opcoes, key, placeholder, on_change=None):
    return st.selectbox(
        rotulo,
        options=[None] + list(opcoes),
        format_func=lambda x: placeholder if x is None else x,
        key=key,
        on_change=on_change,
        placeholder=placeholder,
    )


def seletor_geografico(prefixo, municipios, distritos, bairros, profundidade=4):
    """Selects em cascata Estado → Município → Distrito → Bairro.

    Retorna (uf, municipio, distrito, bairro, cod_mun)."""
    k = lambda nome: f"{prefixo}_{nome}"
    filhos = [k("mun"), k("dist"), k("bairro")]

    estados = sorted(municipios["state"].dropna().unique())
    uf = select_com_placeholder(
        "Estado", estados, k("uf"), "Todos os estados", on_change=_reset_keys(*filhos)
    )

    municipio = distrito = bairro = cod_mun = None

    if uf and profundidade >= 2:
        nomes_mun = (
            municipios.loc[municipios["state"] == uf, "name"]
            .dropna().sort_values().unique()
        )
        municipio = select_com_placeholder(
            "Município", nomes_mun, k("mun"), "Todos os municípios",
            on_change=_reset_keys(*filhos[1:]),
        )

    if municipio:
        info = municipios[(municipios["state"] == uf) & (municipios["name"] == municipio)]
        cod_mun = info["CD_MUN"].iloc[0] if len(info) else None

    if cod_mun and profundidade >= 3:
        nomes_dist = (
            distritos.loc[distritos["CD_MUN"] == cod_mun, "NM_DIST"]
            .dropna().sort_values().unique()
        )
        if len(nomes_dist):
            distrito = select_com_placeholder(
                "Distrito", nomes_dist, k("dist"), "Todos os distritos",
                on_change=_reset_keys(filhos[2]),
            )
        else:
            st.caption("Este município não possui distritos cadastrados.")

    if distrito and profundidade >= 4:
        col_nome = coluna_nome_bairro(bairros)
        nomes_bairro = (
            bairros.loc[
                (bairros["NM_UF"] == uf)
                & (bairros["NM_MUN"] == municipio)
                & (bairros["NM_DIST"] == distrito),
                col_nome,
            ].dropna().sort_values().unique()
        )
        if len(nomes_bairro):
            bairro = select_com_placeholder(
                "Bairro", nomes_bairro, k("bairro"), "Todos os bairros"
            )
        else:
            st.caption("Sem bairros cadastrados para este distrito.")

    return uf, municipio, distrito, bairro, cod_mun


ESPEC_COLUNAS = {
    "nome": ("texto", None),
    "state": ("texto", "UF"),
    "total_population": ("inteiro", "População total"),
    "elderly_total": ("inteiro", "Pop. 70+"),
    "pct_70": ("percentual", "% 70+"),
    "avg_income": ("moeda", "Renda média (R$)"),
    "income_std": ("moeda", "Desvio renda (R$)"),
    "elders_above_8k": ("inteiro", "Idosos > R$ 8 mil (est.)"),
}


def montar_tabela(df, col_nome, rotulo_nome, incluir_uf=False):
    """Prepara o dataframe de exibição e o column_config correspondente."""
    dados = df.copy()
    dados["pct_70"] = (
        dados["elderly_percentage"] / 100 if "elderly_percentage" in dados.columns else np.nan
    )

    ordem = [col_nome]
    if incluir_uf and "state" in dados.columns:
        ordem.append("state")
    ordem += ["elders_above_8k", "elderly_total"]
    if "total_population" in dados.columns:
        ordem += ["pct_70", "total_population"]
    ordem += ["avg_income", "income_std"]
    ordem = [c for c in ordem if c in dados.columns]

    tabela = dados[ordem].copy()
    for c in tabela.columns:
        if c != col_nome and c != "state":
            tabela[c] = pd.to_numeric(tabela[c], errors="coerce")
    # Valores monetários arredondados: centavos não agregam na análise
    for c in ("avg_income", "income_std"):
        if c in tabela.columns:
            tabela[c] = tabela[c].round(0)
    tabela = tabela.sort_values("elders_above_8k", ascending=False, na_position="last")

    renomear, config = {}, {}
    for c in ordem:
        tipo, rotulo = ESPEC_COLUNAS.get(c if c != col_nome else "nome", ("texto", None))
        rotulo = rotulo or (rotulo_nome if c == col_nome else c)
        renomear[c] = rotulo
        if tipo == "inteiro":
            config[rotulo] = st.column_config.NumberColumn(rotulo, format="localized")
        elif tipo == "moeda":
            config[rotulo] = st.column_config.NumberColumn(rotulo, format="localized")
        elif tipo == "percentual":
            config[rotulo] = st.column_config.NumberColumn(rotulo, format="percent")

    return tabela.rename(columns=renomear), config


def botao_download(df, nome_arquivo, key):
    csv = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        "⬇️ Baixar CSV (Excel)",
        data=csv,
        file_name=nome_arquivo,
        mime="text/csv",
        key=key,
        help="Separador ponto e vírgula e decimal com vírgula — abre direto no Excel brasileiro.",
    )


def grafico_ranking(df, col_nome, titulo, maximo=15, minimo=3):
    """Barras horizontais com o top N por potencial de mercado."""
    top = df.nlargest(maximo, "elders_above_8k")[[col_nome, "elders_above_8k"]].copy()
    top.columns = ["nome", "valor"]
    top = top[top["valor"] > 0]
    if len(top) < minimo:
        return
    top["rotulo"] = top["valor"].map(fmt_int)

    base = alt.Chart(top).encode(
        y=alt.Y(
            "nome:N",
            sort="-x",
            title=None,
            axis=alt.Axis(labelColor="#4B5563", labelLimit=190, ticks=False, domainOpacity=0),
        ),
        x=alt.X("valor:Q", title=None, axis=None, scale=alt.Scale(domainMax=float(top["valor"].max()) * 1.18)),
    )
    barras = base.mark_bar(
        color=COR_GRAFICO, size=18, cornerRadiusTopRight=4, cornerRadiusBottomRight=4
    ).encode(
        tooltip=[
            alt.Tooltip("nome:N", title="Local"),
            alt.Tooltip("rotulo:N", title="Idosos 70+ com renda > R$ 8 mil"),
        ]
    )
    rotulos = base.mark_text(align="left", dx=6, color="#374151", fontSize=12).encode(
        text="rotulo:N"
    )

    st.markdown(f"##### {titulo}")
    st.altair_chart(
        (barras + rotulos).properties(height=alt.Step(30)).configure_view(strokeOpacity=0),
        width="stretch",
    )


def exibir_metricas(df, rotulo_unidade):
    total_70 = df["elderly_total"].sum()
    com_renda = df["avg_income"].dropna()
    if len(com_renda):
        pesos = df.loc[df["avg_income"].notna(), "elderly_total"].fillna(1)
        renda_pond = np.average(com_renda, weights=pesos)
    else:
        renda_pond = 0
    total_8k = df["elders_above_8k"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("População 70+", fmt_int(total_70), help="Soma da população com 70 anos ou mais (IBGE V01041).")
    c2.metric("Renda média (pond.)", fmt_brl(renda_pond), help="Renda média do responsável pelo domicílio, ponderada pela população 70+ de cada local.")
    c3.metric("Idosos > R$ 8 mil (est.)", fmt_int(total_8k), help="Estimativa de idosos 70+ em domicílios cujo responsável ganha acima de R$ 8.000/mês (classe B+). Proxy do mercado para ILPIs de alto padrão.")
    c4.metric(rotulo_unidade, fmt_int(len(df)))


# ---------------------------------------------------------------------------
# Aba 1 — Exploração geográfica
# ---------------------------------------------------------------------------
def aba_exploracao(municipios, distritos, bairros, selecao, filtros):
    uf, municipio, distrito, bairro, cod_mun = selecao
    min_pop, min_renda = filtros
    col_bairro = coluna_nome_bairro(bairros)
    aviso = None

    # Resolve o recorte conforme o nível selecionado
    if bairro:
        dados = bairros[
            (bairros["NM_UF"] == uf)
            & (bairros["NM_MUN"] == municipio)
            & (bairros["NM_DIST"] == distrito)
            & (bairros[col_bairro] == bairro)
        ].copy()
        col_nome, rotulo_nome, unidade = col_bairro, "Bairro", "Bairros"
        titulo = f"Bairro {bairro}"
        trilha = ["Brasil", uf, municipio, distrito, bairro]
    elif distrito:
        dados = bairros[
            (bairros["NM_UF"] == uf)
            & (bairros["NM_MUN"] == municipio)
            & (bairros["NM_DIST"] == distrito)
        ].copy()
        col_nome, rotulo_nome, unidade = col_bairro, "Bairro", "Bairros"
        titulo = f"Bairros de {distrito}"
        trilha = ["Brasil", uf, municipio, distrito]
        if dados.empty and cod_mun:
            # O IBGE não mapeou bairros em todos os municípios — mostra o próprio distrito
            dados = distritos[
                (distritos["CD_MUN"] == cod_mun) & (distritos["NM_DIST"] == distrito)
            ].copy()
            col_nome, rotulo_nome, unidade = "NM_DIST", "Distrito", "Distritos"
            titulo = f"Distrito {distrito}"
            aviso = (
                "O Censo 2022 não tem bairros mapeados para este distrito — "
                "exibindo os dados do distrito como um todo."
            )
    elif municipio and cod_mun:
        dados = distritos[distritos["CD_MUN"] == cod_mun].copy()
        col_nome, rotulo_nome, unidade = "NM_DIST", "Distrito", "Distritos"
        titulo = f"Distritos de {municipio}"
        trilha = ["Brasil", uf, municipio]
    elif uf:
        dados = municipios[municipios["state"] == uf].copy()
        col_nome, rotulo_nome, unidade = "name", "Município", "Municípios"
        titulo = f"Municípios de {uf}"
        trilha = ["Brasil", uf]
    else:
        dados = municipios.copy()
        col_nome, rotulo_nome, unidade = "name", "Município", "Municípios"
        titulo = "Todos os municípios do Brasil"
        trilha = ["Brasil"]

    st.caption(" › ".join(trilha))
    st.subheader(titulo)
    if aviso:
        st.info(aviso)

    if dados.empty:
        st.warning("Não há dados para a área selecionada.")
        return

    # Filtros da barra lateral
    total_antes = len(dados)
    if min_pop > 0:
        dados = dados[pd.to_numeric(dados["elderly_total"], errors="coerce").fillna(0) >= min_pop]
    if min_renda > 0:
        dados = dados[pd.to_numeric(dados["avg_income"], errors="coerce").fillna(0) >= min_renda]
    if len(dados) < total_antes:
        st.caption(
            f"Filtros ativos: exibindo **{fmt_int(len(dados))}** de {fmt_int(total_antes)} locais."
        )

    if dados.empty:
        st.info("Nenhum local atende aos filtros definidos. Ajuste os valores na barra lateral.")
        return

    exibir_metricas(dados, unidade)
    st.write("")

    if len(dados) >= 3:
        grafico_ranking(
            dados, col_nome,
            f"Top {min(15, len(dados))} · Idosos 70+ com renda acima de R$ 8 mil",
        )
        st.write("")

    tabela, config = montar_tabela(dados, col_nome, rotulo_nome, incluir_uf=(rotulo_nome == "Município"))
    st.dataframe(tabela, width="stretch", hide_index=True, column_config=config)
    botao_download(tabela, f"ilpi_{rotulo_nome.lower()}s.csv", key="dl_exploracao")


# ---------------------------------------------------------------------------
# Aba 2 — Comparação de locais
# ---------------------------------------------------------------------------
def aba_comparacao(municipios, distritos, bairros):
    if "comparacao" not in st.session_state:
        st.session_state.comparacao = []

    st.markdown("Monte uma lista de locais — de níveis iguais ou diferentes — e compare os indicadores lado a lado.")

    nivel = st.segmented_control(
        "Adicionar por nível",
        options=["Município", "Distrito", "Bairro"],
        default="Município",
        key="comp_nivel",
    ) or "Município"
    profundidade = {"Município": 2, "Distrito": 3, "Bairro": 4}[nivel]

    with st.container(border=True):
        colunas = st.columns(profundidade)
        uf, municipio, distrito, bairro, cod_mun = _seletor_em_colunas(
            colunas, municipios, distritos, bairros, profundidade
        )

        alvo = {"Município": municipio, "Distrito": distrito, "Bairro": bairro}[nivel]
        if st.button("➕ Adicionar à comparação", disabled=alvo is None, key="comp_add"):
            item = _montar_item(nivel, uf, municipio, distrito, bairro, cod_mun,
                                municipios, distritos, bairros)
            if item is None:
                st.warning("Não foi possível localizar os dados deste local.")
            elif any(i["display_name"] == item["display_name"] for i in st.session_state.comparacao):
                st.info(f"**{item['display_name']}** já está na comparação.")
            else:
                st.session_state.comparacao.append(item)

    itens = st.session_state.comparacao
    if not itens:
        st.caption("Nenhum local adicionado ainda.")
        return

    st.write("")
    st.markdown(f"##### Comparando {len(itens)} {'local' if len(itens) == 1 else 'locais'}")

    nomes = [i["display_name"] for i in itens]
    mantidos = st.multiselect(
        "Locais na comparação (clique no ✕ de um item para removê-lo)",
        options=nomes,
        default=nomes,
    )
    if len(mantidos) != len(itens):
        st.session_state.comparacao = [i for i in itens if i["display_name"] in mantidos]
        st.rerun()

    df_comp = pd.DataFrame(itens)
    tabela, config = montar_tabela(df_comp, "display_name", "Local")
    st.dataframe(tabela, width="stretch", hide_index=True, column_config=config)

    if len(itens) >= 2:
        grafico_ranking(
            df_comp, "display_name", "Idosos 70+ com renda acima de R$ 8 mil",
            maximo=len(itens), minimo=2,
        )

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("🗑️ Limpar lista", key="comp_clear"):
            st.session_state.comparacao = []
            st.rerun()
    with c2:
        botao_download(tabela, "ilpi_comparacao.csv", key="dl_comparacao")


def _seletor_em_colunas(colunas, municipios, distritos, bairros, profundidade):
    """Distribui o seletor em cascata pelas colunas já criadas."""
    uf = municipio = distrito = bairro = cod_mun = None
    estados = sorted(municipios["state"].dropna().unique())

    with colunas[0]:
        uf = select_com_placeholder(
            "Estado", estados, "comp_uf", "Selecione...",
            on_change=_reset_keys("comp_mun", "comp_dist", "comp_bairro"),
        )
    if uf and profundidade >= 2:
        nomes = municipios.loc[municipios["state"] == uf, "name"].dropna().sort_values().unique()
        with colunas[1]:
            municipio = select_com_placeholder(
                "Município", nomes, "comp_mun", "Selecione...",
                on_change=_reset_keys("comp_dist", "comp_bairro"),
            )
    if municipio:
        info = municipios[(municipios["state"] == uf) & (municipios["name"] == municipio)]
        cod_mun = info["CD_MUN"].iloc[0] if len(info) else None
    if cod_mun and profundidade >= 3:
        nomes = distritos.loc[distritos["CD_MUN"] == cod_mun, "NM_DIST"].dropna().sort_values().unique()
        with colunas[2]:
            distrito = select_com_placeholder(
                "Distrito", nomes, "comp_dist", "Selecione...",
                on_change=_reset_keys("comp_bairro"),
            )
    if distrito and profundidade >= 4:
        col_nome = coluna_nome_bairro(bairros)
        nomes = bairros.loc[
            (bairros["NM_UF"] == uf)
            & (bairros["NM_MUN"] == municipio)
            & (bairros["NM_DIST"] == distrito),
            col_nome,
        ].dropna().sort_values().unique()
        with colunas[3]:
            bairro = select_com_placeholder("Bairro", nomes, "comp_bairro", "Selecione...")
    return uf, municipio, distrito, bairro, cod_mun


def _montar_item(nivel, uf, municipio, distrito, bairro, cod_mun,
                 municipios, distritos, bairros):
    if nivel == "Município":
        linhas = municipios[(municipios["state"] == uf) & (municipios["name"] == municipio)]
        nome = f"{municipio} · {uf}"
    elif nivel == "Distrito":
        linhas = distritos[(distritos["CD_MUN"] == cod_mun) & (distritos["NM_DIST"] == distrito)]
        nome = f"{distrito} · {municipio} · {uf}"
    else:
        col_nome = coluna_nome_bairro(bairros)
        linhas = bairros[
            (bairros["NM_UF"] == uf)
            & (bairros["NM_MUN"] == municipio)
            & (bairros["NM_DIST"] == distrito)
            & (bairros[col_nome] == bairro)
        ]
        nome = f"{bairro} · {distrito} · {municipio} · {uf}"

    if linhas.empty:
        return None
    linha = linhas.iloc[0]
    item = {
        "display_name": nome,
        "nivel": nivel,
        "elderly_total": float(pd.to_numeric(linha.get("elderly_total"), errors="coerce") or 0),
        "avg_income": float(pd.to_numeric(linha.get("avg_income"), errors="coerce") or 0),
        "income_std": float(pd.to_numeric(linha.get("income_std"), errors="coerce") or 0),
        "elders_above_8k": float(pd.to_numeric(linha.get("elders_above_8k"), errors="coerce") or 0),
    }
    if nivel == "Município":
        item["total_population"] = float(pd.to_numeric(linha.get("total_population"), errors="coerce") or 0)
        item["elderly_percentage"] = float(pd.to_numeric(linha.get("elderly_percentage"), errors="coerce") or 0)
    return item


# ---------------------------------------------------------------------------
# Metodologia
# ---------------------------------------------------------------------------
def secao_metodologia():
    st.write("")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📚 Fontes de dados e metodologia"):
            st.markdown(
                """
                **População idosa (70+)**
                - Fonte: Censo Demográfico IBGE 2022, variável V01041 (contagem oficial de pessoas com 70 anos ou mais).
                - Nos municípios, o valor é agregado a partir dos distritos e conferido contra a população total do Censo.

                **Renda**
                - Renda média mensal do responsável pelo domicílio, por recorte geográfico.
                - A variância da renda é usada para medir dispersão e estimar a cauda alta da distribuição.

                **Cobertura**
                - Todos os 5.570 municípios brasileiros, com detalhamento por distrito e bairro onde o IBGE disponibiliza.
                """
            )
    with col2:
        with st.expander("🧮 Como estimamos “idosos com renda acima de R$ 8 mil”"):
            st.markdown(
                """
                A renda domiciliar costuma seguir uma **distribuição log-normal**. A partir da média e do
                desvio-padrão da renda do responsável em cada local, estimamos a fração de domicílios com
                renda acima de **R$ 8.000/mês** (classe B+) e aplicamos essa fração à população 70+:

                1. **Coeficiente de variação** = desvio-padrão ÷ renda média (limitado a 100%, pois a
                   variância do IBGE às vezes produz valores irreais);
                2. Parâmetros da log-normal calculados a partir da média e do desvio;
                3. Probabilidade de renda > R$ 8 mil obtida pela distribuição acumulada;
                4. **Estimativa final** = população 70+ × probabilidade.

                **Leitura**: quanto maior o valor, maior o público potencial para serviços de longa
                permanência de padrão mais alto. Trate como *proxy* de mercado, não como contagem exata.
                """
            )
    st.caption(
        "Fonte: IBGE, Censo Demográfico 2022 · Ferramenta de apoio à decisão para localização de ILPIs."
    )


# ---------------------------------------------------------------------------
# Aplicação
# ---------------------------------------------------------------------------
def main():
    try:
        with st.spinner("Carregando dados do Censo IBGE 2022..."):
            municipios, distritos, bairros = montar_bases()
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return

    with st.sidebar:
        st.markdown("### 🏡 Localização de ILPIs")
        st.caption("População 70+ e renda · Censo IBGE 2022")
        st.write("")
        st.markdown("**Navegação geográfica**")
        selecao = seletor_geografico("nav", municipios, distritos, bairros)

        st.divider()
        st.markdown("**Filtros**")
        min_pop = st.number_input(
            "População 70+ mínima", min_value=0, value=0, step=100, key="f_pop",
            help="Oculta locais com menos idosos 70+ do que o valor definido.",
        )
        min_renda = st.number_input(
            "Renda média mínima (R$)", min_value=0.0, value=0.0, step=500.0, key="f_renda",
            help="Oculta locais com renda média do responsável abaixo do valor definido.",
        )

        st.write("")
        st.button(
            "🔄 Limpar seleção e filtros",
            width="stretch",
            on_click=_limpar_tudo,
            key="btn_limpar",
        )

    st.title("🏡 Análise de Localização para ILPIs")
    st.markdown(
        "Potencial de mercado para **Instituições de Longa Permanência para Idosos** em todo o Brasil — "
        "navegue por estado, município, distrito e bairro usando a barra lateral."
    )
    st.write("")

    tab_explorar, tab_comparar = st.tabs(["🗺️  Exploração geográfica", "⚖️  Comparar locais"])
    with tab_explorar:
        aba_exploracao(municipios, distritos, bairros, selecao, (min_pop, min_renda))
    with tab_comparar:
        aba_comparacao(municipios, distritos, bairros)

    secao_metodologia()


if __name__ == "__main__":
    main()
