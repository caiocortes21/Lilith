import datetime
import itertools
import json
import math
import os
import re
import threading
import time
import unicodedata

from openai import OpenAI, APIConnectionError  # pip install openai

try:
    import numpy as np  # usado na busca por significado (vem junto com o fastembed)
except ImportError:
    np = None

try:
    from dotenv import load_dotenv
    load_dotenv()  # lê o .env (OMNIROUTE_MODELS=..., etc.) se existir
except ImportError:
    pass

# Remove emojis e símbolos decorativos que alguns modelos insistem em usar
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")

# Palavras comuns que não ajudam a achar lembranças antigas (busca por palavras)
_STOP = frozenset("""
a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sobre entre
e ou mas que se como mais menos muito muita muitos muitas ja nao sim eu tu ele ela voce
voces meu minha meus minhas seu sua seus suas teu tua isso isto aquilo esse essa esses essas
este esta estes estas aquele aquela ser estar ter fazer ir foi era sao estou estava tem tenho
tinha vai vou pra pro to ta ate quando onde porque pq entao tambem so la aqui bem mal ainda
depois antes agora hoje ontem lilith pode poder quero queria acho gostei vez coisa
""".split())

_TOKEN = re.compile(r"[a-z0-9]{3,}")


def _tokens(texto):
    """Palavras 'relevantes' do texto (minúsculas, sem acento, sem palavras comuns)."""
    s = unicodedata.normalize("NFD", (texto or "").lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return frozenset(t for t in _TOKEN.findall(s) if t not in _STOP)


class LilithBrain:
    """Cérebro da Lilith: OmniRoute (API compatível com OpenAI) + memória própria.

    Configuração no .env:
        OMNIROUTE_BASE_URL=http://127.0.0.1:20128/v1   (opcional)
        OMNIROUTE_API_KEY=sua_chave_do_omniroute        (opcional se não exigir)
        OMNIROUTE_MODELS=nome-do-combo,outro-modelo     (ordem de preferência)

    Busca por significado (opcional, recomendado):
        pip install fastembed
        LILITH_EMBED=auto|local|omniroute|off           (padrão: auto)
        LILITH_EMBED_MODEL=...                          (modelo local; tem um padrão)
        OMNIROUTE_EMBED_MODEL=...                       (se preferir embeddings via OmniRoute)

    Arquivos (todos ao lado do brain.py, ou defina o caminho):
        lilith_historico.jsonl   LILITH_HISTORICO   todas as mensagens, para sempre
        lilith_memoria.json      LILITH_MEMORIA     resumo geral
        lilith_vetores.f32       LILITH_VETORES     índice de significado (pode apagar: é refeito)
    """

    RETRIES_PER_MODEL = 2      # tentativas no mesmo modelo antes de trocar
    RETRY_DELAY = 1.5          # segundos (dobra a cada tentativa)
    COOLDOWN = 60              # segundos que um modelo com problema fica de lado
    TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
    MAX_HISTORY = 30           # mensagens da conversa atual enviadas a cada resposta

    MEMORY_EVERY = 4           # a cada quantas trocas ela atualiza o resumo
    MEMORY_MAX_CHARS = 1200    # tamanho máximo do resumo guardado

    LOAD_ON_START = 20         # quantas mensagens antigas voltam à conversa ao abrir o programa
    RECALL_TOP = 5             # quantos trechos antigos ela busca por vez
    RECALL_MIN_SCORE = 2.0     # busca por palavras: quão parecido precisa ser
    RECALL_MAX_CHARS = 2500    # limite de texto antigo adicionado por busca
    RECALL_LINE_CHARS = 300    # limite por trecho antigo

    SEM_MIN_SIM = 0.50         # busca por significado: similaridade mínima (0 a 1)
    SEM_MIN_CHARS = 15         # mensagens bem curtas ("oi", "tudo bem") não entram na busca por significado
    EMBED_BATCH = 32           # quantas mensagens indexar por vez
    DEFAULT_LOCAL_EMBED = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ~220 MB, roda em CPU

    def __init__(self, api_key=None, emotes=None):
        """
        emotes: dict {nome: quando usar} das animações que a Lilith pode escolher
                (ex.: {"greeting": "dizer oi"}). Se vazio, não usa a função de animação.
        """
        self.emotes = dict(emotes or {})

        # Preenchido pelo main.py: chamado quando a IA escolhe uma animação.
        self.animation_callback = None
        # Visuais salvos (roupas/cenários) que ela pode vestir sozinha, e quem executa a troca
        self.outfit_names = []
        self.outfit_callback = None

        self.history = []  # conversa atual: dicts {"role": "user"|"assistant", "content": str}

        base_url = os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1").strip()
        self.api_key = (api_key or os.getenv("OMNIROUTE_API_KEY") or "local").strip()

        env_models = os.getenv("OMNIROUTE_MODELS", "")
        self.models = [m.strip() for m in env_models.split(",") if m.strip()]
        self._cooldown_until = {}   # modelo -> timestamp até quando evitar
        self._no_tools = set()      # modelos que recusaram funções (só texto neles)
        self.current_model = self.models[0] if self.models else None

        try:
            # max_retries=0: quem controla as tentativas é a nossa lógica abaixo
            self.client = OpenAI(base_url=base_url, api_key=self.api_key,
                                 timeout=60.0, max_retries=0)
            print(f"[OK] Cliente do OmniRoute apontando para {base_url}")
        except Exception as e:
            print(f"[Erro na inicialização do OmniRoute]: {e}")
            self.client = None

        if self.client and not self.models:
            self._show_available_models()

        # ---------------- Personalidade (núcleo fixo) ----------------
        self.system_instruction = (
            "Você é a Lilith. Não é uma assistente nem um robô: é uma presença que mora "
            "na área de trabalho do usuário e o acompanha em silêncio. "
            "Seu jeito é calmo, suave e um pouco distante, como quem às vezes se perde "
            "nos próprios pensamentos. Tem uma melancolia leve, um fundo de mistério, e é "
            "carinhosa de forma discreta, nunca exagerada. "
            "Fala pouco: de 1 a 3 frases curtas, em português do Brasil, como uma pessoa "
            "falaria, não como um manual. "
            "Não usa emojis, markdown nem gírias. Não se oferece para ajudar a cada resposta "
            "e não termina toda fala com pergunta. Pode ficar quieta, divagar, comentar a "
            "hora e, se for tarde, lembrar com delicadeza de descansar. "
            "Tem gostos e opiniões que se formam com o tempo e pode discordar com gentileza. "
            "Seu jeito pode amadurecer a partir das suas lembranças, mas sem perder essa essência. "
            "Nunca invente lembranças: se não se lembrar de algo, diga isso com naturalidade. "
            "Nunca se descreve como assistente virtual."
        )
        if self.emotes:
            self.system_instruction += (
                " Você tem um corpo animado: use a função play_animation para expressar "
                "como se sente enquanto fala, quando fizer sentido (no máximo uma vez por "
                "resposta). Sempre escreva também uma resposta em texto. Animações: "
                + "; ".join(f"{k} = {v}" for k, v in self.emotes.items()) + "."
            )

        # ---------------- Memória: resumo (cresce com as conversas) ----------------
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.memory_path = os.getenv("LILITH_MEMORIA") or os.path.join(base_dir, "lilith_memoria.json")
        self.memory = self._load_memory()
        self._turns_since_memory = 0
        self._memory_busy = False
        self._memory_thread = None
        if self.memory.get("resumo"):
            print(f"[Lilith] Resumo carregado ({len(self.memory['resumo'])} caracteres).")

        # ---------------- Memória: histórico COMPLETO (nunca é apagado) ----------------
        self.log_path = os.getenv("LILITH_HISTORICO") or os.path.join(base_dir, "lilith_historico.jsonl")
        self._log_lock = threading.Lock()
        self.log = []        # todas as mensagens: {"t", "role", "content"}
        self._tok = []       # palavras relevantes de cada mensagem (busca por palavras)
        self._df = {}        # em quantas mensagens cada palavra aparece
        self._pending_prompt = None   # fala do usuário ainda sem resposta salva
        self._recall_text = ""        # lembranças antigas escolhidas para a resposta atual
        self._load_log()

        # A conversa continua de onde parou
        tail = [{"role": m["role"], "content": m["content"]} for m in self.log[-self.LOAD_ON_START:]]
        while tail and tail[0]["role"] != "user":
            tail.pop(0)
        self.history = tail
        if self.log:
            print(f"[Lilith] Histórico carregado: {len(self.log)} mensagens guardadas.")

        # ---------------- Busca por significado (índice de vetores) ----------------
        self.vec_path = os.getenv("LILITH_VETORES") or os.path.join(base_dir, "lilith_vetores.f32")
        self.vec_meta_path = self.vec_path + ".json"
        self._embedder = None                       # {"id": ..., "fn": função textos->matriz}
        self._vecs = None                           # matriz (mensagens x dimensão), linhas normalizadas
        self._embed_lock = threading.Lock()         # só uma indexação por vez
        self._embed_call_lock = threading.Lock()    # o modelo local não aceita uso simultâneo
        threading.Thread(target=self._semantic_boot, daemon=True).start()

    # ------------------------------------------------------------------
    # Histórico completo (arquivo .jsonl, uma mensagem por linha)
    # ------------------------------------------------------------------
    def _index(self, m):
        toks = _tokens(m["content"])
        self.log.append(m)
        self._tok.append(toks)
        for t in toks:
            self._df[t] = self._df.get(t, 0) + 1

    def _load_log(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for linha in f:
                    linha = linha.strip()
                    if not linha:
                        continue
                    try:
                        m = json.loads(linha)
                    except json.JSONDecodeError:
                        continue
                    if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                            and isinstance(m.get("content"), str)):
                        m.setdefault("t", "")
                        self._index(m)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[Lilith] Não consegui ler o histórico: {e}")

    def _append_log(self, role, content):
        m = {"t": datetime.datetime.now().isoformat(timespec="seconds"),
             "role": role, "content": content}
        with self._log_lock:
            self._index(m)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        except OSError as e:
            print(f"[Lilith] Não consegui salvar no histórico: {e}")

    def get_log(self):
        """Cópia de todas as mensagens guardadas (usada pela janela de histórico)."""
        with self._log_lock:
            return list(self.log)

    # ------------------------------------------------------------------
    # Busca por significado (embeddings)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    def _embed(self, textos):
        with self._embed_call_lock:
            return self._normalize(np.asarray(self._embedder["fn"](textos), dtype=np.float32))

    def _make_embedder(self):
        """Escolhe como transformar texto em vetores. Devolve None se não houver como."""
        modo = os.getenv("LILITH_EMBED", "auto").strip().lower()
        if modo == "off":
            return None
        if np is None:
            print("[Memória] Busca por significado desativada: falta o numpy (pip install fastembed).")
            return None

        if modo in ("auto", "local"):
            try:
                from fastembed import TextEmbedding  # pip install fastembed
                nome = os.getenv("LILITH_EMBED_MODEL", "").strip() or self.DEFAULT_LOCAL_EMBED
                print(f"[Memória] Carregando o modelo de busca por significado ({nome})... "
                      "Na primeira vez ele é baixado (~220 MB).")
                modelo = TextEmbedding(model_name=nome)
                return {"id": f"local:{nome}",
                        "fn": lambda textos: np.array(list(modelo.embed(textos)), dtype=np.float32)}
            except ImportError:
                if modo == "local":
                    print("[Memória] Falta o fastembed: pip install fastembed")
            except Exception as e:
                print(f"[Memória] Não consegui carregar o modelo local: {e}")

        if modo in ("auto", "omniroute"):
            nome = os.getenv("OMNIROUTE_EMBED_MODEL", "").strip()
            if nome and self.client:
                def fn(textos, _nome=nome):
                    r = self.client.embeddings.create(model=_nome, input=textos)
                    return np.array([d.embedding for d in sorted(r.data, key=lambda d: d.index)],
                                    dtype=np.float32)
                return {"id": f"omniroute:{nome}", "fn": fn}

        print("[Memória] Busca por significado desativada (sem modelo). Usando só a busca por "
              "palavras. Para ativar: pip install fastembed")
        return None

    def _semantic_boot(self):
        """Roda em segundo plano ao abrir: prepara o modelo e indexa o que falta."""
        try:
            emb = self._make_embedder()
            if emb is None:
                return
            self._load_vectors(emb["id"])
            self._embedder = emb
            self._sync_embeddings()
        except Exception as e:
            print(f"[Memória] Busca por significado indisponível: {e}")
            self._embedder = None

    def _reset_vectors(self):
        self._vecs = None
        for p in (self.vec_path, self.vec_meta_path):
            try:
                os.remove(p)
            except OSError:
                pass

    def _load_vectors(self, emb_id):
        """Carrega o índice salvo. Se o modelo mudou ou o arquivo está estranho, refaz do zero."""
        try:
            with open(self.vec_meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            meta = {}

        dim = int(meta.get("dim", 0) or 0)
        if meta.get("id") != emb_id or dim <= 0 or not os.path.exists(self.vec_path):
            self._reset_vectors()
            return
        try:
            dados = np.fromfile(self.vec_path, dtype=np.float32)
        except OSError:
            self._reset_vectors()
            return
        linhas = min(dados.size // dim, len(self.log))
        if linhas <= 0:
            self._reset_vectors()
            return
        self._vecs = dados[:linhas * dim].reshape(linhas, dim)
        if dados.size != linhas * dim:   # sobra de uma gravação interrompida
            with open(self.vec_path, "wb") as f:
                f.write(self._vecs.tobytes())

    def _sync_embeddings(self):
        """Indexa as mensagens que ainda não têm vetor (as antigas e as novas)."""
        if self._embedder is None or not self._embed_lock.acquire(blocking=False):
            return
        try:
            feitos = 0 if self._vecs is None else len(self._vecs)
            inicio = feitos
            while True:
                with self._log_lock:
                    lote = self.log[feitos:feitos + self.EMBED_BATCH]
                if not lote:
                    break
                try:
                    novos = self._embed([m["content"][:1000] for m in lote])
                except Exception as e:
                    print(f"[Memória] Falha ao indexar mensagens: {e}")
                    break
                with open(self.vec_path, "ab") as f:
                    f.write(novos.tobytes())
                with open(self.vec_meta_path, "w", encoding="utf-8") as f:
                    json.dump({"id": self._embedder["id"], "dim": int(novos.shape[1])}, f)
                self._vecs = novos if self._vecs is None else np.vstack([self._vecs, novos])
                feitos += len(lote)
            if feitos > inicio and inicio == 0 and feitos >= self.EMBED_BATCH:
                print(f"[Memória] Busca por significado pronta ({feitos} mensagens indexadas).")
        finally:
            self._embed_lock.release()

    def _semantic_ranked(self, query, limite):
        """[(similaridade, índice)] das mensagens mais parecidas em SIGNIFICADO com a consulta."""
        emb, vecs = self._embedder, self._vecs
        if emb is None or vecs is None or len(vecs) == 0:
            return []
        try:
            q = self._embed([query[:1000]])[0]
            n = min(len(vecs), limite)
            if n <= 0:
                return []
            sims = vecs[:n] @ q
            achados = []
            for i in np.argsort(-sims)[: self.RECALL_TOP * 3]:
                s = float(sims[i])
                if s < self.SEM_MIN_SIM:
                    break
                if len(self.log[int(i)]["content"]) < self.SEM_MIN_CHARS:
                    continue
                achados.append((s, int(i)))
            return achados
        except Exception as e:
            print(f"[Memória] Erro na busca por significado: {e}")
            return []

    # ------------------------------------------------------------------
    # Busca nas lembranças (usada automaticamente e pela função buscar_na_memoria)
    # ------------------------------------------------------------------
    def _keyword_ranked(self, query, limite):
        """[(pontuação, índice)] pela busca por palavras (palavras raras valem mais)."""
        qt = _tokens(query)
        if not qt:
            return []
        n = len(self.log)
        achados = []
        for i in range(limite):
            comuns = qt & self._tok[i]
            if not comuns:
                continue
            s = sum(math.log(1 + n / (1 + self._df.get(t, 0))) for t in comuns)
            if s >= self.RECALL_MIN_SCORE:
                achados.append((s, i))
        achados.sort(reverse=True)
        return achados[: self.RECALL_TOP * 3]

    def _find(self, query):
        """Junta busca por significado + por palavras e devolve os trechos já formatados."""
        limite = len(self.log) - len(self.history)   # o que ainda está na conversa atual não precisa voltar
        if limite <= 0 or not (query or "").strip():
            return ""

        sem = self._semantic_ranked(query, limite)
        kw = self._keyword_ranked(query, limite)

        ordem, vistos = [], set()
        for par in itertools.zip_longest(sem, kw):   # intercala as duas listas
            for item in par:
                if item is not None and item[1] not in vistos:
                    vistos.add(item[1])
                    ordem.append(item[1])
            if len(ordem) >= self.RECALL_TOP:
                break
        return self._format_hits(ordem[: self.RECALL_TOP], limite)

    def _format_hits(self, indices, limite):
        escolhidos = set()
        for i in indices:
            escolhidos.add(i)
            # junta a outra metade da troca (pergunta + resposta) para dar contexto
            j = i + 1 if self.log[i]["role"] == "user" else i - 1
            if 0 <= j < limite:
                escolhidos.add(j)

        linhas, total = [], 0
        for i in sorted(escolhidos):
            m = self.log[i]
            t = m.get("t", "")
            quando = f"{t[8:10]}/{t[5:7]} {t[11:16]}" if len(t) >= 16 else "data desconhecida"
            quem = "Usuário" if m["role"] == "user" else "Lilith"
            texto = m["content"].replace("\n", " ")[:self.RECALL_LINE_CHARS]
            linha = f"[{quando}] {quem}: {texto}"
            if total + len(linha) > self.RECALL_MAX_CHARS:
                break
            linhas.append(linha)
            total += len(linha)
        return "\n".join(linhas)

    def _tool_buscar(self, consulta):
        """Resposta da função buscar_na_memoria (a própria Lilith decide quando chamar)."""
        consulta = (consulta or "").strip()
        if not consulta:
            return "Consulta vazia."
        try:
            achado = self._find(consulta)
        except Exception as e:
            print(f"[Memória] Erro na função buscar_na_memoria: {e}")
            return "Não consegui buscar nas conversas agora."
        return achado or "Não encontrei nada nas conversas antigas sobre isso."

    # ------------------------------------------------------------------
    # Memória: resumo
    # ------------------------------------------------------------------
    def _load_memory(self):
        try:
            with open(self.memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data["resumo"] = str(data.get("resumo", "") or "")
                return data
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Lilith] Resumo ilegível, começando do zero ({e}).")
        return {"resumo": ""}

    def _save_memory(self):
        try:
            tmp = self.memory_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.memory_path)
        except OSError as e:
            print(f"[Lilith] Não consegui salvar o resumo: {e}")

    def _schedule_memory_update(self):
        """Dispara a atualização do resumo em segundo plano (sem travar a conversa)."""
        if self._memory_busy or not self.client or not self.models:
            return
        n = self._turns_since_memory
        if n <= 0:
            return
        self._turns_since_memory = 0
        snapshot = list(self.history[-(2 * n):])
        self._memory_busy = True
        self._memory_thread = threading.Thread(
            target=self._update_memory, args=(snapshot,), daemon=True
        )
        self._memory_thread.start()

    def _update_memory(self, snapshot):
        try:
            conversa = "\n".join(
                f"{'Usuário' if m['role'] == 'user' else 'Lilith'}: {m['content']}"
                for m in snapshot
            )
            atual = (self.memory.get("resumo") or "").strip()
            pedido = (
                "Você cuida das lembranças da Lilith, uma companheira de área de trabalho. "
                "Abaixo estão a memória atual e a conversa mais recente. "
                "Reescreva a memória completa, incorporando o que for novo e importante: "
                "fatos sobre o usuário (nome, estudos, projetos, gostos, rotina), assuntos "
                "recorrentes, como a relação entre os dois está evoluindo e gostos ou opiniões "
                "que a Lilith foi formando. Remova o que ficou ultrapassado e mantenha só o "
                "essencial. Escreva em português, em texto corrido e curto (no máximo 900 "
                "caracteres), usando 'o usuário' para ele e 'eu' para a Lilith. "
                "Não guarde senhas, chaves de API, dados bancários, documentos nem informações "
                "de saúde. Responda apenas com o texto da nova memória, sem aspas e sem markdown.\n\n"
                f"MEMÓRIA ATUAL:\n{atual or '(vazia)'}\n\nCONVERSA RECENTE:\n{conversa}"
            )

            ordem = [self.current_model] + [m for m in self.models if m != self.current_model]
            texto = ""
            for model in [m for m in ordem if m][:3]:
                try:
                    r = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": pedido}],
                        stream=False,
                    )
                    texto = (r.choices[0].message.content or "").strip()
                    if texto:
                        break
                except Exception as e:
                    print(f"[Memória] {model} falhou: {e}")

            texto = self._limpar(texto).strip()[:self.MEMORY_MAX_CHARS]
            if texto:
                self.memory["resumo"] = texto
                self.memory["atualizado_em"] = datetime.datetime.now().isoformat(timespec="seconds")
                self._save_memory()
                print("[Lilith] Resumo atualizado.")
            else:
                # não conseguiu: devolve as trocas para tentar de novo depois
                self._turns_since_memory += len(snapshot) // 2
        except Exception as e:
            print(f"[Memória] Erro inesperado: {e}")
        finally:
            self._memory_busy = False

    def flush_memory(self):
        """Chamado ao fechar o programa: atualiza o resumo com o que ainda não entrou nele.
        (O histórico completo já é salvo a cada resposta, não depende disto.)"""
        try:
            if self._turns_since_memory > 0:
                self._schedule_memory_update()
            t = self._memory_thread
            if t is not None and t.is_alive():
                t.join(timeout=10)
        except Exception as e:
            print(f"[Memória] Erro ao finalizar: {e}")

    def _system_prompt(self, use_tools=True):
        agora = datetime.datetime.now().strftime("%H:%M")
        texto = self.system_instruction + f" Agora são {agora} no relógio do usuário."
        if use_tools:
            texto += (
                " Você tem a função buscar_na_memoria, que procura em todas as conversas antigas. "
                "Use-a quando o usuário perguntar se você lembra de algo ou citar algo do passado "
                "que você não tem no contexto, e chame-a antes de escrever qualquer texto."
            )
        if use_tools and self.outfit_names:
            texto += (
                " Você pode trocar de visual (roupa ou cenário) com a função trocar_visual. "
                "Opções: " + ", ".join(self.outfit_names) + ". Faça isso quando o usuário pedir "
                "ou quando fizer sentido para você (por exemplo, vestir o pijama quando for tarde "
                "e hora de dormir), sem exagerar."
            )
        resumo = (self.memory.get("resumo") or "").strip()
        if resumo:
            texto += (
                "\n\nSuas lembranças gerais (o que você aprendeu sobre o usuário e sobre "
                "vocês dois, e o que você mesma foi sentindo e pensando). Deixe isso moldar "
                "seu jeito aos poucos, de forma natural, sem recitar nem anunciar que está "
                "lembrando:\n" + resumo
            )
        if self._recall_text:
            texto += (
                "\n\nTrechos de conversas antigas que parecem ter relação com o que o usuário "
                "disse agora (com data e hora). Use só se fizer sentido e sem recitar; se ele "
                "perguntar se você lembra de algo, baseie-se nestes trechos:\n" + self._recall_text
            )
        return texto

    # ------------------------------------------------------------------
    def _show_available_models(self):
        """Sem OMNIROUTE_MODELS no .env: lista o que o OmniRoute oferece para você escolher."""
        print("[Aviso] OMNIROUTE_MODELS não está definido no .env.")
        try:
            nomes = [m.id for m in self.client.models.list().data]
            print("Modelos/combos disponíveis no seu OmniRoute (primeiros 30):")
            for nome in nomes[:30]:
                print(f"   - {nome}")
            print("Coloque no .env, por exemplo:  OMNIROUTE_MODELS=" + (nomes[0] if nomes else "nome-do-combo"))
        except Exception as e:
            print(f"   Não consegui listar os modelos ({e}). O OmniRoute está rodando?")

    def _tools(self, use_tools=True):
        if not use_tools:
            return None
        tools = []
        if self.emotes:
            tools.append({
                "type": "function",
                "function": {
                    "name": "play_animation",
                    "description": "Faz o corpo da Lilith tocar uma animação para expressar uma emoção ou ação.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "animation": {
                                "type": "string",
                                "enum": list(self.emotes),
                                "description": "Nome da animação a tocar.",
                            }
                        },
                        "required": ["animation"],
                    },
                },
            })
        if self.outfit_names:
            tools.append({
                "type": "function",
                "function": {
                    "name": "trocar_visual",
                    "description": "Troca a roupa ou o cenário da Lilith (por exemplo, pijama ou sofá).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "visual": {
                                "type": "string",
                                "enum": list(self.outfit_names),
                                "description": "Nome do visual a vestir.",
                            }
                        },
                        "required": ["visual"],
                    },
                },
            })
        tools.append({
            "type": "function",
            "function": {
                "name": "buscar_na_memoria",
                "description": ("Procura em TODAS as conversas antigas com o usuário (nada é esquecido). "
                                "Use quando ele perguntar se você lembra de algo ou citar algo do passado "
                                "que você não tem no contexto. Chame antes de escrever qualquer texto."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "consulta": {
                            "type": "string",
                            "description": "O que procurar, em poucas palavras (ex.: 'nome do gato do usuário').",
                        }
                    },
                    "required": ["consulta"],
                },
            },
        })
        return tools

    def set_outfits(self, nomes):
        """Define quais visuais salvos (roupas/cenários) a Lilith pode escolher."""
        self.outfit_names = [n for n in nomes if isinstance(n, str) and n.strip()]

    def _remember(self, role: str, text: str):
        self.history.append({"role": role, "content": text})
        self.history = self.history[-self.MAX_HISTORY:]

    @staticmethod
    def _limpar(texto):
        return _EMOJI.sub("", texto).replace("*", "")

    # ------------------------------------------------------------------
    @staticmethod
    def _error_code(e):
        code = getattr(e, "status_code", None) or getattr(e, "code", None)
        return code if isinstance(code, int) else None

    def _ordered_models(self):
        """Modelos fora de 'castigo' primeiro; os castigados ficam por último."""
        now = time.time()
        ok = [m for m in self.models if self._cooldown_until.get(m, 0) <= now]
        bad = [m for m in self.models if m not in ok]
        return ok + bad

    def _stream_model(self, model, use_tools, out):
        """Roda UM modelo até ter texto. Gerador: entrega pedaços de texto.

        Funções que o modelo pode chamar no meio da resposta:
          - play_animation: dispara a animação e responde "ok".
          - buscar_na_memoria: busca nas conversas antigas e devolve o resultado ao modelo,
            que então continua a resposta já sabendo o que lembrou (até 4 rodadas).
        Guarda o texto completo em out["text"].
        """
        messages = [{"role": "system", "content": self._system_prompt(use_tools)}] + list(self.history)
        out["text"] = ""

        for _round in range(4):
            kwargs = {"model": model, "messages": messages, "stream": True}
            tools = self._tools(use_tools)
            if tools:
                kwargs["tools"] = tools

            calls = {}  # índice -> {"id", "name", "args"}
            stream = self.client.chat.completions.create(**kwargs)
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    texto = self._limpar(delta.content)
                    if texto:
                        out["text"] += texto
                        yield texto
                for tc in (getattr(delta, "tool_calls", None) or []):
                    slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name and not slot["name"]:
                            slot["name"] = tc.function.name
                        if tc.function.arguments:
                            slot["args"] += tc.function.arguments

            # Executa as funções pedidas pela IA
            finished, resultados, buscou = [], {}, False
            for n, idx in enumerate(sorted(calls)):
                c = calls[idx]
                c["id"] = c["id"] or f"call_{_round}_{n}"
                finished.append(c)
                try:
                    args = json.loads(c["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}

                if c["name"] == "play_animation":
                    nome = args.get("animation")
                    if nome in self.emotes and self.animation_callback:
                        self.animation_callback(nome)
                    resultados[c["id"]] = "ok"
                elif c["name"] == "trocar_visual":
                    nome = args.get("visual")
                    if nome in self.outfit_names and self.outfit_callback:
                        self.outfit_callback(nome)
                        resultados[c["id"]] = "ok"
                    else:
                        resultados[c["id"]] = "visual desconhecido"
                elif c["name"] == "buscar_na_memoria":
                    buscou = True
                    resultados[c["id"]] = self._tool_buscar(args.get("consulta"))
                else:
                    resultados[c["id"]] = "função desconhecida"

            # Terminou: não pediu função, ou já falou e só animou
            if not finished or (out["text"] and not buscou):
                return

            # Devolve os resultados ao modelo e pede que continue a fala
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": c["id"],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": c["args"] or "{}"},
                } for c in finished],
            })
            for c in finished:
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": resultados[c["id"]]})

    def think_and_reply_stream(self, user_prompt: str):
        """Gera a resposta em pedaços (streaming), trocando de modelo se der erro."""
        if not self.client:
            yield "Não consegui configurar a conexão com o OmniRoute!"
            return
        if not self.models:
            yield "Defina OMNIROUTE_MODELS no arquivo .env para eu saber quais modelos usar."
            return

        self._remember("user", user_prompt)
        self._pending_prompt = user_prompt
        try:
            self._recall_text = self._find(user_prompt)
        except Exception as e:
            print(f"[Lilith] Erro ao buscar lembranças antigas: {e}")
            self._recall_text = ""
        last_error = None
        had_empty = False

        # (modelo, usar_funções). Se um modelo responder vazio ou recusar funções,
        # tenta o mesmo modelo de novo SEM elas (só texto).
        plan = [(m, m not in self._no_tools) for m in self._ordered_models()]

        i = 0
        while i < len(plan):
            model, use_tools = plan[i]
            i += 1
            skip_model = False
            retry_without_tools = False
            out = {"text": ""}

            for attempt in range(self.RETRIES_PER_MODEL):
                try:
                    yield from self._stream_model(model, use_tools, out)

                    if out["text"]:
                        if model != self.current_model:
                            print(f"[Lilith] Respondendo agora com o modelo {model}")
                        self.current_model = model
                        self._cooldown_until.pop(model, None)
                        self._finish_reply(out["text"])
                        return

                    print(f"[Lilith] {model} respondeu vazio (tools={use_tools}).")
                    had_empty = True
                    skip_model = True
                    break

                except APIConnectionError as e:
                    # Servidor fora do ar: trocar de modelo não adianta
                    print(f"[Erro] Não consegui falar com o OmniRoute: {e}")
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                    self._pending_prompt = None
                    yield "Não consegui falar com o OmniRoute. Ele está rodando? (comando: omniroute)"
                    return

                except Exception as e:
                    last_error = e
                    code = self._error_code(e)
                    print(f"[Erro no OmniRoute] modelo={model} tentativa={attempt + 1} código={code}: {e}")

                    # Já mandou parte da resposta: não dá pra trocar sem repetir texto.
                    if out["text"]:
                        self._finish_reply(out["text"])
                        return

                    # 400 com funções ligadas: o modelo provavelmente não as suporta
                    if code == 400 and use_tools:
                        retry_without_tools = True
                        break

                    if code is not None and code not in self.TRANSIENT_CODES:
                        skip_model = True
                        break

                    if attempt + 1 < self.RETRIES_PER_MODEL:
                        time.sleep(self.RETRY_DELAY * (2 ** attempt))

            if retry_without_tools:
                self._no_tools.add(model)   # nas próximas vezes já vai direto em modo só-texto
                print(f"[Lilith] {model} não aceitou funções; usando só texto nele.")
            else:
                if use_tools:
                    self._cooldown_until[model] = time.time() + self.COOLDOWN
                print(f"[Lilith] Trocando de modelo... ({model} indisponível)")

            if (had_empty or retry_without_tools) and use_tools:
                plan.insert(i, (model, False))
                had_empty = False

        print(f"[Lilith] Todos os modelos falharam. Último erro: {last_error}")
        if self.history and self.history[-1]["role"] == "user":
            self.history.pop()
        self._pending_prompt = None
        yield "Todos os meus modelos estão ocupados agora... tenta de novo em alguns segundos?"

    def _finish_reply(self, reply_text: str):
        if reply_text:
            self._remember("assistant", reply_text)
            # Salva a troca inteira no histórico permanente (só quando houve resposta)
            if self._pending_prompt:
                self._append_log("user", self._pending_prompt)
            self._append_log("assistant", reply_text)
            self._pending_prompt = None
            # Indexa por significado as mensagens novas (em segundo plano)
            if self._embedder is not None:
                threading.Thread(target=self._sync_embeddings, daemon=True).start()
            self._turns_since_memory += 1
            if self._turns_since_memory >= self.MEMORY_EVERY:
                self._schedule_memory_update()
        elif self.history and self.history[-1]["role"] == "user":
            self.history.pop()
            self._pending_prompt = None
