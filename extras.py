"""Extras da Lilith: catálogo do modelo, roupas/cenários salvos, explorador e passeio pelo desktop.

Este módulo só depende do PyQt6 básico (não do navegador embutido).
"""
import json
import os
import random
import re
import time

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QVBoxLayout, QWidget,
)


# ----------------------------------------------------------------------
# Catálogo: lista tudo que o modelo tem, num arquivo de texto
# ----------------------------------------------------------------------
def _grupo_anim(nome):
    partes = nome.split("_")
    return "_".join(partes[:2]) if len(partes) >= 3 else partes[0]


def write_catalog(path, animations, skins):
    """Salva todas as animações e skins do modelo, agrupadas, em um .txt. Devolve o caminho."""
    linhas = ["LILITH - catálogo do modelo",
              f"{len(animations)} animações, {len(skins)} skins", "",
              "=== ANIMAÇÕES (agrupadas pelo começo do nome) ==="]
    grupos = {}
    for n in animations:
        grupos.setdefault(_grupo_anim(n), []).append(n)
    for g in sorted(grupos):
        linhas.append(f"\n[{g}] ({len(grupos[g])})")
        linhas += [f"  {n}" for n in sorted(grupos[g])]

    linhas += ["", "", "=== SKINS (roupas, cenários, acessórios) ==="]
    sk = {}
    for n in skins:
        sk.setdefault(n.split("/")[0], []).append(n)
    for g in sorted(sk):
        linhas.append(f"\n[{g}] ({len(sk[g])})")
        linhas += [f"  {n}" for n in sorted(sk[g])]
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas) + "\n")
        return path
    except OSError as e:
        print(f"[Lilith] Não consegui salvar o catálogo: {e}")
        return None


# ----------------------------------------------------------------------
# Roupas e cenários salvos (lilith_roupas.json)
# ----------------------------------------------------------------------
class Presets:
    """Visuais salvos: cada um é {"skins": [...], "anim": nome-ou-None}.
    Ex.: "pijama rosa" = skins do pijama; "sofá azul" = skins do sofá + animação de sentar."""

    def __init__(self, path):
        self.path = path
        self.data = {"atual": None, "presets": {}, "andar": {}, "passear_sozinha": True}
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as e:
            print(f"[Lilith] lilith_roupas.json ilegível, começando do zero ({e}).")
            return
        if not isinstance(raw, dict):
            return

        for nome, v in (raw.get("presets") or {}).items():
            if isinstance(v, list):
                v = {"skins": v}
            if isinstance(v, dict):
                skins = [s for s in (v.get("skins") or []) if isinstance(s, str)]
                anim = v.get("anim") if isinstance(v.get("anim"), str) else None
                self.data["presets"][str(nome)] = {"skins": skins, "anim": anim}

        atual = raw.get("atual")
        if isinstance(atual, str) and atual in self.data["presets"]:
            self.data["atual"] = atual
        andar = raw.get("andar")
        if isinstance(andar, dict):
            self.data["andar"] = {k: v for k, v in andar.items()
                                  if k in ("left", "right", "neutral") and isinstance(v, str)}
        if isinstance(raw.get("passear_sozinha"), bool):
            self.data["passear_sozinha"] = raw["passear_sozinha"]

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"[Lilith] Não consegui salvar lilith_roupas.json: {e}")

    def names(self):
        return sorted(self.data["presets"], key=str.lower)

    def get(self, nome):
        return self.data["presets"].get(nome)

    def put(self, nome, skins, anim=None):
        self.data["presets"][nome] = {"skins": list(skins), "anim": anim}
        self.save()

    def delete(self, nome):
        self.data["presets"].pop(nome, None)
        if self.data["atual"] == nome:
            self.data["atual"] = None
        self.save()

    def set_current(self, nome):
        self.data["atual"] = nome
        self.save()

    def set_walk(self, lado, nome):
        self.data["andar"][lado] = nome
        self.save()

    def set_auto_walk(self, valor):
        self.data["passear_sozinha"] = bool(valor)
        self.save()


# ----------------------------------------------------------------------
# Passeio pelo desktop
# ----------------------------------------------------------------------
_WALK = re.compile(r"(^|[_/])(walk|run|stroll)", re.I)
_NOT_WALK = re.compile(r"sit|sleep|lie|lay|couch|sofa", re.I)


def _lado(nome):
    n = nome.lower()
    if re.search(r"(^|[_/])left([_/]|$)", n):
        return "left"
    if re.search(r"(^|[_/])right([_/]|$)", n):
        return "right"
    return None


def _melhor(lista):
    if not lista:
        return None
    return sorted(lista, key=lambda n: ("casual" not in n.lower(), len(n), n))[0]


def detect_walk(animations):
    """Tenta achar sozinho as animações de andar pelo nome. Devolve {left, right, neutral}."""
    cand = [n for n in animations if _WALK.search(n) and not _NOT_WALK.search(n)]
    return {
        "left": _melhor([n for n in cand if _lado(n) == "left"]),
        "right": _melhor([n for n in cand if _lado(n) == "right"]),
        "neutral": _melhor([n for n in cand if _lado(n) is None]),
    }


class WalkController:
    """Faz a janela da Lilith andar de um lado a outro da tela, com a animação de andar."""
    TICK_MS = 16
    SPEED = 90.0               # pixels por segundo (ajuste se parecer que ela desliza)
    MIN_WAIT, MAX_WAIT = 20, 60   # segundos parada entre um passeio e outro
    FIRST_WAIT = 15            # segundos até o primeiro passeio
    MIN_DIST = 150             # distância mínima de cada passeio (pixels)
    EDGE_MARGIN = 60           # quanto a janela pode passar da borda da tela
    NEUTRAL_FACES = "right"    # se só existe UMA animação de andar sem lado: para onde ela olha?

    def __init__(self, overlay, animator, presets):
        self.o = overlay
        self.a = animator
        self.presets = presets
        self.auto = bool(presets.data.get("passear_sozinha", True))
        self.animations = set()
        self.detected = {}
        self.names = {"left": None, "right": None, "neutral": None}
        self.walking = False
        self._dir = 1
        self._target = 0.0
        self._x = 0.0
        self._last = 0.0
        self._next = time.monotonic() + self.FIRST_WAIT
        self._warned = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)

    # --- configuração --------------------------------------------------
    def set_catalog(self, animations):
        self.animations = set(animations)
        self.detected = detect_walk(animations)
        self.refresh_names()

    def refresh_names(self):
        cfg = self.presets.data.get("andar") or {}
        for k in ("left", "right", "neutral"):
            escolhido = cfg.get(k)
            if escolhido not in self.animations:
                escolhido = self.detected.get(k)
            self.names[k] = escolhido
        if self.available():
            print(f"[Lilith] Animações de andar: {self.names}")
        elif not self._warned:
            self._warned = True
            print("[Lilith] Não achei animação de andar pelo nome. Abra 'Explorar animações e "
                  "roupas' na bandeja e marque qual usar.")

    def available(self):
        return any(self.names.values())

    def set_auto(self, valor):
        self.auto = bool(valor)
        self.presets.set_auto_walk(self.auto)
        if not self.auto and self.walking:
            self._finish()

    def start(self):
        if self.available() and not self.timer.isActive():
            self.timer.start(self.TICK_MS)

    def shutdown(self):
        self.timer.stop()

    # --- movimento -----------------------------------------------------
    def _geo(self):
        scr = QApplication.screenAt(self.o.frameGeometry().center()) or QApplication.primaryScreen()
        return scr.availableGeometry()

    def _anim_for(self, lado):
        n = self.names
        if n.get(lado):
            return n[lado], False
        oposto = "left" if lado == "right" else "right"
        if n.get(oposto):
            return n[oposto], True          # usa a do outro lado, espelhada
        if n.get("neutral"):
            return n["neutral"], lado != self.NEUTRAL_FACES
        return None, False

    def walk_to(self, x=None):
        """Começa a andar até x (ou até um ponto aleatório). Devolve True se começou."""
        if self.walking or not self.available():
            return False
        geo = self._geo()
        lo = geo.left() - self.EDGE_MARGIN
        hi = geo.right() - self.o.width() + self.EDGE_MARGIN
        atual = float(self.o.x())
        if hi - lo < self.MIN_DIST:
            return False
        if x is None:
            for _ in range(10):
                x = random.uniform(lo, hi)
                if abs(x - atual) >= self.MIN_DIST:
                    break
            else:
                x = lo if atual - lo > hi - atual else hi
        x = max(lo, min(hi, x))
        lado = "right" if x > atual else "left"
        anim, espelhar = self._anim_for(lado)
        if not anim:
            return False
        self._dir = 1 if lado == "right" else -1
        self._target = x
        self._x = atual
        self.a.flip(espelhar)
        self.a.body(anim, True, False)
        self.walking = True
        self._last = time.monotonic()
        return True

    def _finish(self):
        self.walking = False
        self.a.flip(False)
        self.a.go_idle()
        self._next = time.monotonic() + random.uniform(self.MIN_WAIT, self.MAX_WAIT)

    def stop_walking(self):
        if self.walking:
            self._finish()

    def _tick(self):
        try:
            agora = time.monotonic()
            if self.walking:
                if self.o.is_busy():            # clicou, arrastou, falou...
                    self._finish()
                    return
                dt = min(0.1, agora - self._last)
                self._last = agora
                self._x += self._dir * self.SPEED * dt
                chegou = (self._dir > 0 and self._x >= self._target) or \
                         (self._dir < 0 and self._x <= self._target)
                if chegou:
                    self._x = self._target
                self.o.move(int(self._x), self.o.y())
                if chegou:
                    self._finish()
            elif self.auto and agora >= self._next and not self.o.is_busy():
                self.walk_to(None)
        except Exception as e:
            print(f"[Lilith] Erro no passeio: {e}")


# ----------------------------------------------------------------------
# Explorador: ver/tocar todas as animações, montar e salvar roupas e cenários
# ----------------------------------------------------------------------
def _filtrar(lista, texto):
    t = texto.strip().lower()
    for i in range(lista.count()):
        it = lista.item(i)
        it.setHidden(bool(t) and t not in it.text().lower())


class ExplorerWindow(QWidget):
    """ctx = a janela principal (precisa de: apply_preset, reset_scene, presets_changed, walker)."""

    def __init__(self, animator, presets, ctx):
        super().__init__()
        self.a = animator
        self.presets = presets
        self.ctx = ctx
        self.setWindowTitle("Lilith - animações e roupas")
        self.resize(760, 680)
        self.setStyleSheet("""
            QWidget { background-color: #14141a; color: #eaeaf0;
                      font-family: 'Segoe UI', sans-serif; font-size: 12px; }
            QListWidget, QLineEdit, QComboBox { background-color: #1d1d26; border: 1px solid #33333f;
                                                padding: 3px; }
            QPushButton { background-color: #2a2a38; border: 1px solid #44445a; padding: 5px 8px; }
            QPushButton:hover { background-color: #36364a; }
            QLabel#dica { color: #8a8a9c; }
            QLabel#status { color: #b79cff; }
        """)

        raiz = QVBoxLayout(self)
        topo = QHBoxLayout()
        raiz.addLayout(topo, 1)

        # ---- coluna da esquerda: animações ----
        esq = QVBoxLayout()
        topo.addLayout(esq, 1)
        esq.addWidget(QLabel("Animações  (duplo clique para tocar)"))
        self.anim_filter = QLineEdit()
        self.anim_filter.setPlaceholderText("filtrar (ex.: walk, sit, sleep)")
        self.anim_list = QListWidget()
        self.loop_chk = QCheckBox("repetir (loop)")
        esq.addWidget(self.anim_filter)
        esq.addWidget(self.anim_list, 1)
        esq.addWidget(self.loop_chk)

        esq.addWidget(QLabel("Usar a animação selecionada para ANDAR:"))
        linha_andar = QHBoxLayout()
        for texto, lado in (("→ direita", "right"), ("← esquerda", "left"), ("sem lado", "neutral")):
            b = QPushButton(texto)
            b.clicked.connect(lambda _=False, l=lado: self._set_walk(l))
            linha_andar.addWidget(b)
        esq.addLayout(linha_andar)
        self.walk_lbl = QLabel()
        self.walk_lbl.setObjectName("dica")
        self.walk_lbl.setWordWrap(True)
        esq.addWidget(self.walk_lbl)

        linha_acoes = QHBoxLayout()
        b_passear = QPushButton("Passear agora")
        b_passear.clicked.connect(self._passear)
        b_normal = QPushButton("Parar / voltar ao normal")
        b_normal.clicked.connect(self._reset)
        linha_acoes.addWidget(b_passear)
        linha_acoes.addWidget(b_normal)
        esq.addLayout(linha_acoes)

        # ---- coluna da direita: skins ----
        dire = QVBoxLayout()
        topo.addLayout(dire, 1)
        dire.addWidget(QLabel("Skins: roupas, cenários, acessórios  (marque e aplique)"))
        self.skin_filter = QLineEdit()
        self.skin_filter.setPlaceholderText("filtrar (ex.: pajama, sofa)")
        self.skin_list = QListWidget()
        dire.addWidget(self.skin_filter)
        dire.addWidget(self.skin_list, 1)
        dica = QLabel("Dica: marque 'default' + uma opção de cada grupo (roupa, óculos, "
                      "chapéu...), como no exemplo do animator.py.")
        dica.setObjectName("dica")
        dica.setWordWrap(True)
        dire.addWidget(dica)
        linha_skin = QHBoxLayout()
        b_aplicar = QPushButton("Aplicar marcadas")
        b_aplicar.clicked.connect(self._apply_marked)
        b_limpar = QPushButton("Desmarcar tudo")
        b_limpar.clicked.connect(self._uncheck_all)
        linha_skin.addWidget(b_aplicar)
        linha_skin.addWidget(b_limpar)
        dire.addLayout(linha_skin)

        # ---- rodapé: visuais salvos ----
        raiz.addWidget(QLabel("Visuais salvos (a Lilith também pode escolher entre eles sozinha)"))
        salvos = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(180)
        b_usar = QPushButton("Usar")
        b_usar.clicked.connect(self._apply_preset)
        b_del = QPushButton("Excluir")
        b_del.clicked.connect(self._delete_preset)
        salvos.addWidget(self.combo, 1)
        salvos.addWidget(b_usar)
        salvos.addWidget(b_del)
        raiz.addLayout(salvos)

        novo = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("nome do visual (ex.: pijama rosa, sofá azul, normal)")
        self.scene_chk = QCheckBox("incluir a animação selecionada (ex.: sentada)")
        b_salvar = QPushButton("Salvar marcadas como visual")
        b_salvar.clicked.connect(self._save_preset)
        novo.addWidget(self.name_edit, 1)
        novo.addWidget(self.scene_chk)
        novo.addWidget(b_salvar)
        raiz.addLayout(novo)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        raiz.addWidget(self.status)

        self.anim_filter.textChanged.connect(lambda t: _filtrar(self.anim_list, t))
        self.skin_filter.textChanged.connect(lambda t: _filtrar(self.skin_list, t))
        self.anim_list.itemDoubleClicked.connect(self._play_item)

    # --- dados ---------------------------------------------------------
    def set_catalog(self, animations, skins):
        self.anim_list.clear()
        for n in animations:
            self.anim_list.addItem(n)
        self.skin_list.clear()
        for n in skins:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Unchecked)
            self.skin_list.addItem(it)
        self._refresh_combo()
        self._refresh_walk_label()
        self.status.setText(f"{len(animations)} animações e {len(skins)} skins carregadas.")

    def _refresh_combo(self):
        atual = self.combo.currentText()
        self.combo.clear()
        self.combo.addItems(self.presets.names())
        i = self.combo.findText(atual)
        if i >= 0:
            self.combo.setCurrentIndex(i)

    def _refresh_walk_label(self):
        n = self.ctx.walker.names
        self.walk_lbl.setText(
            f"Andar: direita = {n.get('right') or '-'} | esquerda = {n.get('left') or '-'} | "
            f"sem lado = {n.get('neutral') or '-'}")

    def _marked_skins(self):
        return [self.skin_list.item(i).text() for i in range(self.skin_list.count())
                if self.skin_list.item(i).checkState() == Qt.CheckState.Checked]

    # --- ações ---------------------------------------------------------
    def _play_item(self, item):
        self.ctx.walker.stop_walking()
        loop = self.loop_chk.isChecked()
        self.a.body(item.text(), loop, not loop)
        self.status.setText(f"Tocando: {item.text()}")

    def _set_walk(self, lado):
        it = self.anim_list.currentItem()
        if it is None:
            self.status.setText("Selecione uma animação na lista primeiro.")
            return
        self.presets.set_walk(lado, it.text())
        self.ctx.walker.refresh_names()
        self.ctx.walker.start()
        self._refresh_walk_label()
        self.status.setText(f"'{it.text()}' será usada para andar ({lado}).")

    def _passear(self):
        if not self.ctx.walker.walk_to(None):
            self.status.setText("Ainda não há animação de andar escolhida (ou não há espaço na tela).")

    def _reset(self):
        self.ctx.reset_scene()
        self.status.setText("Voltou ao normal.")

    def _apply_marked(self):
        skins = self._marked_skins()
        if not skins:
            self.status.setText("Marque ao menos uma skin.")
            return
        self.a.set_outfit(skins)
        self.status.setText(f"Aplicadas {len(skins)} skins.")

    def _uncheck_all(self):
        for i in range(self.skin_list.count()):
            self.skin_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _save_preset(self):
        nome = self.name_edit.text().strip()
        skins = self._marked_skins()
        anim = None
        if self.scene_chk.isChecked() and self.anim_list.currentItem() is not None:
            anim = self.anim_list.currentItem().text()
        if not nome:
            self.status.setText("Dê um nome ao visual.")
            return
        if not skins and not anim:
            self.status.setText("Marque skins (e/ou escolha uma animação) antes de salvar.")
            return
        self.presets.put(nome, skins, anim)
        self._refresh_combo()
        i = self.combo.findText(nome)
        if i >= 0:
            self.combo.setCurrentIndex(i)
        self.ctx.presets_changed()
        self.status.setText(f"Visual '{nome}' salvo ({len(skins)} skins"
                            + (f" + animação {anim}" if anim else "") + ").")

    def _apply_preset(self):
        nome = self.combo.currentText()
        if nome:
            self.ctx.apply_preset(nome)
            self.status.setText(f"Visual '{nome}' aplicado.")

    def _delete_preset(self):
        nome = self.combo.currentText()
        if nome:
            self.presets.delete(nome)
            self._refresh_combo()
            self.ctx.presets_changed()
            self.status.setText(f"Visual '{nome}' excluído.")

    def open(self):
        self._refresh_combo()
        self._refresh_walk_label()
        self.show()
        self.raise_()
        self.activateWindow()
