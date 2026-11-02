"""Ponte Python -> JavaScript para controlar a Lilith (Spine 4.2).

O modelo funciona em camadas (trilhas) que tocam ao mesmo tempo:
  trilha 0 = corpo      (stand_casual_*, sit_casual_*, left_/right_casual_*, ...)
  trilha 1 = boca       (talk_happy_*, talk_sad_*)
  trilha 2 = expressão  (emoji_smile_*, emoji_sad_*, emoji_angry_* ...)
  trilha 3 = piscar     (emoji_front_blink, em loop)
"""
import json

from PyQt6.QtWebEngineWidgets import QWebEngineView

# ----------------------------------------------------------------------
# Corpo base
# ----------------------------------------------------------------------
IDLE_BODY = "stand_casual_idle"
THINK_BODY = "stand_casual_think"
THINK_FACE = "emoji_calm_1"
TALK_MOUTH = {"happy": "talk_happy_long", "sad": "talk_sad_long"}

# Estados simples usados pelo main.py
STATES = {
    "idle":     ("body", IDLE_BODY, True),
    "thinking": ("body", THINK_BODY, True),
}

# ----------------------------------------------------------------------
# Emoções que a IA pode escolher (function calling do Gemini).
# nome que a IA vê -> corpo + expressão do rosto.
# As expressões (emoji_*_N) são palpites de qual variação combina:
# se não gostar de uma, troque o número (veja a lista completa abaixo).
# ----------------------------------------------------------------------
EMOTES = {
    "greeting":  {"body": "stand_casual_greet",       "face": "emoji_smile_2"},
    "happy":     {"body": "stand_casual_happy_sigh",  "face": "emoji_smile_1"},
    "laugh":     {"body": "stand_casual_shy_giggle",  "face": "emoji_smile_3"},
    "shy":       {"body": "stand_casual_blush",       "face": "emoji_smile_4"},
    "flirty":    {"body": "stand_casual_flirt",       "face": "emoji_smile_5"},
    "confused":  {"body": "stand_casual_confuse",     "face": "emoji_daze_1"},
    "surprised": {"body": "stand_casual_look_around", "face": "emoji_surprise_1"},
    "sad":       {"body": "stand_casual_look_down",   "face": "emoji_sad_1"},
    "crying":    {"body": "stand_casual_fake_cry",    "face": "emoji_sad_2"},
    "angry":     {"body": "stand_casual_fake_angry",  "face": "emoji_angry_1"},
    "pout":      {"body": "stand_casual_pout",        "face": "emoji_wronged_1"},
    "sleepy":    {"body": "stand_casual_yawn",        "face": "emoji_sleepy_1"},
    "stretch":   {"body": "stand_casual_stretch",     "face": "emoji_calm_1"},
    "clap":      {"body": "stand_casual_clap",        "face": "emoji_smile_2"},
    "wave":      {"body": "stand_casual_lazy_wave",   "face": "emoji_smile_1"},
    "magic":     {"body": "stand_casual_magic_cast",  "face": "emoji_smile_4"},
    "music":     {"body": "stand_casual_music",       "face": "emoji_smile_6"},
    "fear":      {"body": "stand_casual_hide",        "face": "emoji_fear_1"},
}

# Descrição curta de cada emoção, enviada para a IA saber quando usar.
EMOTE_HINTS = {
    "greeting": "cumprimentar / dizer oi", "happy": "alegria tranquila, satisfação",
    "laugh": "rir de leve", "shy": "ficar envergonhada", "flirty": "provocar de forma charmosa",
    "confused": "não entendeu algo", "surprised": "surpresa", "sad": "tristeza leve",
    "crying": "choro dramático", "angry": "bravinha de brincadeira", "pout": "fazer birra / emburrar",
    "sleepy": "sono, bocejo", "stretch": "espreguiçar", "clap": "comemorar / aplaudir",
    "wave": "acenar", "magic": "executar um comando ou 'mágica'",
    "music": "cantarolar / música", "fear": "susto, esconder-se",
}


class LilithAnimator:
    def __init__(self, web_view: QWebEngineView):
        self.web = web_view
        self.idle_body = IDLE_BODY   # corpo de descanso (muda se ela estiver sentada, por ex.)
        self.animations = []         # todas as animações do modelo (preenchido por load_catalog)
        self.skins = []              # todas as skins do modelo (preenchido por load_catalog)

    def _run(self, js: str, callback=None):
        page = self.web.page()
        if callback is None:
            page.runJavaScript(js)
        else:
            page.runJavaScript(js, callback)

    def _call(self, fn: str, *args):
        params = ", ".join(json.dumps(a) for a in args)
        self._run(f"window.lilith && window.lilith.{fn}({params});")

    # --- ações de baixo nível ------------------------------------------
    def body(self, name: str, loop: bool = True, then_idle: bool = True):
        self._call("setBody", name, loop, self.idle_body if then_idle else None)

    def face(self, name: str):
        self._call("setFace", name)

    def reset_face(self):
        self._call("setFace", "emoji_none")

    # --- alto nível ----------------------------------------------------
    def play_state(self, state: str):
        if state == "thinking":
            self._call("setThinking", True, THINK_BODY, self.idle_body, THINK_FACE)
        elif state == "idle":
            self.body(self.idle_body, True, False)

    def start_talking(self, mood: str = "happy"):
        """Boca mexendo; se estava 'pensando', volta o corpo pro idle."""
        self._call("setThinking", False, THINK_BODY, self.idle_body, "emoji_none")
        self._call("setMouth", TALK_MOUTH.get(mood, TALK_MOUTH["happy"]))

    def stop_talking(self):
        self._call("setMouth", None)

    def play_emote(self, emote: str):
        data = EMOTES.get(emote)
        if not data:
            return
        self._call("setThinking", False, THINK_BODY, self.idle_body, "emoji_none")
        self.face(data["face"])
        self.body(data["body"], False, True)

    def set_outfit(self, skins):
        """Ex.: ["default", "default/casual", "default/glasses_none", "default/hat_none"]"""
        self._call("setSkins", list(skins))

    def go_idle(self):
        """Volta para o corpo de descanso (em pé, ou sentada se estiver numa cena)."""
        self.body(self.idle_body, True, False)

    def flip(self, flipped: bool):
        """Espelha a Lilith na horizontal (usado ao andar para o lado contrário)."""
        if flipped:
            js = ("(function(){var e=document.documentElement;"
                  "e.style.transformOrigin=(window.innerWidth/2)+'px 0px';"
                  "e.style.transform='scaleX(-1)';})();")
        else:
            js = "document.documentElement.style.transform='';"
        self._run(js)

    def load_catalog(self, on_ready):
        """Pergunta ao modelo quais animações e skins ele tem.
        Chama on_ready(animacoes, skins); as listas vêm vazias se o modelo ainda não carregou."""
        def got_skins(skins):
            self.skins = sorted(str(x) for x in (skins or []))
            on_ready(self.animations, self.skins)

        def got_anims(anims):
            self.animations = sorted(str(x) for x in (anims or []))
            self.list_skins(got_skins)

        self.list_animations(got_anims)

    def list_animations(self, callback):
        self._run("window.lilith ? window.lilith.listAnimations() : []", callback)

    def list_skins(self, callback):
        self._run("window.lilith ? window.lilith.listSkins() : []", callback)
