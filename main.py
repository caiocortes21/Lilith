import sys
import os
import html
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

from PyQt6.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QFrame, QInputDialog,
    QSystemTrayIcon, QMenu, QTextBrowser
)
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect, QThread, pyqtSignal, QUrl, QEvent
from PyQt6.QtGui import QPixmap, QPainter, QColor, QIcon, QAction, QCursor
from PyQt6.QtWebEngineWidgets import QWebEngineView

from brain import LilithBrain
from animator import LilithAnimator, EMOTES, EMOTE_HINTS, IDLE_BODY
from extras import Presets, ExplorerWindow, WalkController, write_catalog


def get_project_root():
    """Pasta que contém `renderer/` e `assets/` (funciona também empacotado)."""
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        candidates.append(exe_dir)
        candidates.append(os.path.join(exe_dir, "_internal"))
        if hasattr(sys, "_MEIPASS"):
            candidates.append(sys._MEIPASS)
    candidates.append(os.path.dirname(os.path.abspath(__file__)))
    candidates.append(os.path.abspath("."))

    for root in candidates:
        if os.path.exists(os.path.join(root, "renderer", "index.html")):
            return root
    return candidates[-2]


def get_assets_path(relative_subpath=""):
    return os.path.join(get_project_root(), "assets", relative_subpath)


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def start_local_server(root_dir):
    """Serve o projeto em http://127.0.0.1:<porta> (evita problemas de file://)."""
    handler = partial(_QuietHandler, directory=root_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class AIWorker(QThread):
    """Thread em segundo plano que recebe a fala da IA em blocos de texto."""
    chunk_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    animation_signal = pyqtSignal(str)
    outfit_signal = pyqtSignal(str)

    def __init__(self, brain: LilithBrain, prompt: str):
        super().__init__()
        self.brain = brain
        self.prompt = prompt

    def run(self):
        self.brain.animation_callback = self.animation_signal.emit
        self.brain.outfit_callback = self.outfit_signal.emit
        try:
            for chunk in self.brain.think_and_reply_stream(self.prompt):
                self.chunk_signal.emit(chunk)
        except Exception as e:  # rede de segurança: a thread nunca derruba o programa
            print(f"[Erro inesperado na IA]: {e}")
            self.chunk_signal.emit("Ops, tive um pequeno lapso... tenta de novo?")
        finally:
            self.finished_signal.emit()


class SpeechBubble(QFrame):
    """Balão de Diálogo responsivo com streaming em tempo real."""
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 25, 240);
                border: 2px solid rgba(255, 255, 255, 90);
                border-radius: 12px;
                padding: 10px 14px;
            }
            QLabel {
                color: #FFFFFF;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                background: transparent;
                border: none;
                padding: 0px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.label = QLabel("", self)
        self.label.setWordWrap(True)
        self.label.setFixedWidth(230)
        layout.addWidget(self.label)

        self.hide()

        self.hide_timer = QTimer(self)
        self.hide_timer.setSingleShot(True)
        self.hide_timer.timeout.connect(self.hide_bubble)

        self.current_text = ""

    def start_speech(self):
        """Prepara o balão para começar a receber as palavras."""
        self.hide_timer.stop()
        self.current_text = ""
        self.label.setText("")
        self.show()
        self._refit()

    def _refit(self):
        """Recalcula a altura do texto com quebra de linha e ajusta balão e janela.

        QLabel com quebra de linha não informa a altura certa ao layout, e o fim
        do texto ficava cortado. Calculamos a altura na mão.
        """
        width = self.label.width() or self.label.maximumWidth()
        rect = self.label.fontMetrics().boundingRect(
            0, 0, width, 100000,
            int(Qt.TextFlag.TextWordWrap), self.current_text or " "
        )
        self.label.setFixedHeight(rect.height() + 6)
        self.layout().activate()
        self.adjustSize()
        parent = self.parentWidget()
        if parent:
            if callable(parent.layout) and parent.layout() is not None:
                parent.layout().activate()
            parent.adjustSize()

    def append_chunk(self, chunk: str):
        """Acrescenta cada palavra enviada pela IA instantaneamente no balão."""
        self.current_text += chunk
        self.label.setText(self.current_text)
        self._refit()

    def finish_speech(self, duration_ms=9000):
        """Inicia a contagem para sumir o balão após a fala terminar."""
        if duration_ms > 0:
            self.hide_timer.start(duration_ms)

    def hide_bubble(self):
        self.hide()
        if self.parentWidget():
            self.parentWidget().adjustSize()


class HistoryWindow(QWidget):
    """Janela com TODAS as conversas guardadas (para reler o que sumiu do balão)."""
    MAX_SHOW = 500   # quantas mensagens mostrar (o arquivo guarda todas)

    def __init__(self, brain: LilithBrain):
        super().__init__()
        self.brain = brain
        self.setWindowTitle("Histórico da Lilith")
        self.resize(480, 640)
        self.setStyleSheet("""
            QWidget { background-color: #14141a; color: #eaeaf0;
                      font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QTextBrowser { border: none; padding: 6px; }
            QLabel#rodape { color: #7d7d8c; font-size: 11px; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.view = QTextBrowser(self)
        self.view.setOpenLinks(False)
        layout.addWidget(self.view)

        self.footer = QLabel(self)
        self.footer.setObjectName("rodape")
        self.footer.setWordWrap(True)
        self.footer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.footer)

    def refresh(self):
        log = self.brain.get_log()
        total = len(log)
        shown = log[-self.MAX_SHOW:]

        partes = []
        if total > len(shown):
            partes.append(
                f'<p align="center" style="color:#7d7d8c">'
                f'(mostrando as últimas {len(shown)} de {total} mensagens)</p>'
            )
        dia_atual = None
        for m in shown:
            t = m.get("t", "")
            dia = t[:10]
            if dia and dia != dia_atual:
                dia_atual = dia
                partes.append(
                    f'<p align="center" style="color:#7d7d8c">— {dia[8:10]}/{dia[5:7]}/{dia[:4]} —</p>'
                )
            hora = t[11:16]
            if m["role"] == "user":
                quem, cor = "Você", "#9ecbff"
            else:
                quem, cor = "Lilith", "#e3b6ff"
            texto = html.escape(m["content"]).replace("\n", "<br>")
            partes.append(
                f'<p><span style="color:#7d7d8c">{hora}</span> '
                f'<b style="color:{cor}">{quem}:</b> {texto}</p>'
            )
        if not partes:
            partes.append('<p style="color:#7d7d8c">Ainda não conversamos por aqui.</p>')

        self.view.setHtml("".join(partes))
        QTimer.singleShot(0, lambda: self.view.verticalScrollBar().setValue(
            self.view.verticalScrollBar().maximum()
        ))
        self.footer.setText(f"{total} mensagens guardadas em: {self.brain.log_path}")

    def open(self):
        self.refresh()
        self.show()
        self.raise_()
        self.activateWindow()


class LilithOverlay(QWidget):
    # Tamanho da área onde a Lilith é desenhada (ajuste se ela ficar cortada/pequena)
    VIEW_WIDTH = 340
    VIEW_HEIGHT = 550

    # Cliques só na personagem (o resto da janela deixa o clique passar para o que está atrás)
    CLICK_THROUGH = True   # False = volta ao comportamento antigo (retângulo inteiro clicável)
    HIT_ALPHA = 24         # pixel mais transparente que isso conta como "fora da personagem"
    HIT_RADIUS = 4         # tolerância em pixels ao redor do cursor (facilita acertar a borda)
    POLL_MS = 30           # de quanto em quanto tempo checa se o mouse está sobre a personagem
    GRAB_MS = 150          # de quanto em quanto tempo atualiza a silhueta (só com o mouse por perto)

    def __init__(self, scale_factor=0.5):
        super().__init__()

        self.scale_factor = scale_factor
        self.brain = LilithBrain(emotes={k: EMOTE_HINTS.get(k, '') for k in EMOTES})
        self.history_win = HistoryWindow(self.brain)
        self.drag_position = QPoint()
        self.ai_thread = None
        self._first_chunk = True

        # Estado da "área clicável" (só onde a personagem é desenhada)
        self._hit_img = None        # silhueta da última captura (None = tudo clicável)
        self._hit_fail = 0          # quantas capturas inválidas seguidas
        self._hit_diag_done = False
        self._click_through = False # a janela está "invisível" para o mouse agora?
        self._cursor_near = False
        self._last_grab = 0.0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 15, 15, 15)
        self.main_layout.setSpacing(6)

        self.bubble = SpeechBubble(self)
        self.main_layout.addWidget(self.bubble, alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Lilith (Spine) dentro de um navegador embutido transparente ---
        self.server = start_local_server(get_project_root())
        port = self.server.server_address[1]

        self.web = QWebEngineView(self)
        self.web.setFixedSize(self.VIEW_WIDTH, self.VIEW_HEIGHT)
        self.web.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.web.setStyleSheet("background: transparent;")
        self.web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.web.page().setBackgroundColor(Qt.GlobalColor.transparent)
        self.web.loadFinished.connect(self._on_web_loaded)
        self.web.load(QUrl(f"http://127.0.0.1:{port}/renderer/index.html"))
        self.main_layout.addWidget(self.web, alignment=Qt.AlignmentFlag.AlignCenter)

        self.animator = LilithAnimator(self.web)

        # Extras: roupas/cenários salvos, explorador de animações e passeio pelo desktop
        self._base_dir = os.path.dirname(os.path.abspath(__file__))
        self.presets = Presets(os.path.join(self._base_dir, "lilith_roupas.json"))
        self.brain.set_outfits(self.presets.names())
        self.scene_anim = None          # animação de cena ativa (ex.: sentada no sofá)
        self._dialog_open = False
        self._last_interaction = 0.0
        self.walker = WalkController(self, self.animator, self.presets)
        self.explorer = ExplorerWindow(self.animator, self.presets, self)

        # Vigia o cursor: sobre a personagem = clicável; fora dela = o clique passa direto
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_cursor)

        # Ícone na bandeja do sistema (perto do relógio): histórico e sair
        self.tray = self._make_tray()

        self.adjustSize()
        self.position_on_screen()

        QTimer.singleShot(1500, self._welcome_message)

    # ------------------------------------------------------------------
    # Histórico e ícone da bandeja
    # ------------------------------------------------------------------
    def _make_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return None
        pm = QPixmap(32, 32)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#b79cff"))
        p.drawEllipse(3, 3, 26, 26)
        p.end()

        tray = QSystemTrayIcon(QIcon(pm), self)
        tray.setToolTip("Lilith (clique para ver o histórico)")

        menu = QMenu()
        act_hist = QAction("Ver histórico", menu)
        act_hist.triggered.connect(self.open_history)
        act_quit = QAction("Sair", menu)
        act_quit.triggered.connect(QApplication.quit)
        act_expl = QAction("Explorar animações e roupas", menu)
        act_expl.triggered.connect(self.open_explorer)
        act_walk_now = QAction("Passear agora", menu)
        act_walk_now.triggered.connect(lambda: self.walker.walk_to(None))
        act_walk_auto = QAction("Passear sozinha pelo desktop", menu)
        act_walk_auto.setCheckable(True)
        act_walk_auto.setChecked(self.walker.auto)
        act_walk_auto.toggled.connect(self.walker.set_auto)
        menu.addAction(act_hist)
        menu.addAction(act_expl)
        menu.addSeparator()
        menu.addAction(act_walk_now)
        menu.addAction(act_walk_auto)
        menu.addSeparator()
        menu.addAction(act_quit)
        tray.setContextMenu(menu)
        self._tray_menu = menu  # mantém a referência viva

        tray.activated.connect(
            lambda reason: self.open_history()
            if reason == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        tray.show()
        return tray

    def open_history(self):
        self.history_win.open()

    # ------------------------------------------------------------------
    # Extras: explorador, roupas/cenários, catálogo e passeio
    # ------------------------------------------------------------------
    def open_explorer(self):
        self.explorer.open()

    def is_busy(self) -> bool:
        """Ela está ocupada demais para passear? (conversando, arrastada, sentada, em diálogo)"""
        if self._dialog_open or self.scene_anim:
            return True
        t = self.ai_thread
        if t is not None and t.isRunning():
            return True
        if self.bubble.isVisible():
            return True
        return (time.monotonic() - self._last_interaction) < 1.5

    def presets_changed(self):
        """Chamado pelo explorador quando os visuais salvos mudam."""
        self.brain.set_outfits(self.presets.names())

    def reset_scene(self):
        """Volta ao normal: em pé, parada, sem cena."""
        self.scene_anim = None
        self.animator.idle_body = IDLE_BODY
        self.walker.stop_walking()
        self.animator.flip(False)
        self.animator.go_idle()

    def apply_preset(self, nome: str):
        """Veste um visual salvo (roupa e/ou cenário). Também é chamado pela própria IA."""
        p = self.presets.get(nome)
        if not p:
            print(f"[Lilith] Visual desconhecido: {nome}")
            return
        self.walker.stop_walking()
        self.animator.flip(False)
        if p.get("skins"):
            self.animator.set_outfit(p["skins"])
        self.presets.set_current(nome)
        anim = p.get("anim")
        if anim:
            self.scene_anim = anim
            self.animator.idle_body = anim
            self.animator.body(anim, True, False)
        else:
            self.reset_scene()
        print(f"[Lilith] Visual aplicado: {nome}")

    def _load_catalog(self, tentativa=1):
        """Descobre todas as animações e skins do modelo e salva a lista em lilith_catalogo.txt."""
        def pronto(anims, skins):
            if not anims and tentativa < 10:
                QTimer.singleShot(2000, lambda: self._load_catalog(tentativa + 1))
                return
            print(f"[Lilith] Catálogo do modelo: {len(anims)} animações, {len(skins)} skins.")
            caminho = write_catalog(os.path.join(self._base_dir, "lilith_catalogo.txt"), anims, skins)
            if caminho:
                print(f"[Lilith] Lista completa salva em: {caminho}")
            self.explorer.set_catalog(anims, skins)
            self.walker.set_catalog(anims)
            self.walker.start()
            # Volta com o visual que ela estava usando da última vez
            atual = self.presets.data.get("atual")
            if atual and self.presets.get(atual):
                self.apply_preset(atual)

        self.animator.load_catalog(pronto)

    def shutdown(self):
        """Chamado ao fechar o programa."""
        self.walker.shutdown()
        if self.tray is not None:
            self.tray.hide()
        self.brain.flush_memory()

    # ------------------------------------------------------------------
    # Área clicável: só a personagem (e o balão) recebe o mouse.
    # O resto da janela (transparente) deixa o clique passar para o que está atrás.
    #
    # Importante: a janela NÃO é recortada (recortar cortava as pontas do cabelo e
    # piscava). Só ligamos/desligamos o "transparente para o mouse" conforme o
    # cursor está ou não sobre a personagem. O desenho nunca é alterado.
    # ------------------------------------------------------------------
    def _set_click_through(self, enable: bool):
        """Windows: liga/desliga WS_EX_TRANSPARENT (a janela fica 'invisível' para o mouse)."""
        if enable == self._click_through or sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.GetWindowLongW.restype = ctypes.c_long
            user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            user32.SetWindowLongW.restype = ctypes.c_long
            GWL_EXSTYLE, WS_EX_TRANSPARENT = -20, 0x20
            hwnd = int(self.winId())
            estilo = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            novo = (estilo | WS_EX_TRANSPARENT) if enable else (estilo & ~WS_EX_TRANSPARENT)
            if novo != estilo:
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, novo)
            self._click_through = enable
        except Exception as e:
            print(f"[Lilith] Não consegui ajustar o clique-através ({e}); desativado.")
            self.CLICK_THROUGH = False
            self.poll_timer.stop()

    @staticmethod
    def _button_down() -> bool:
        """Algum botão do mouse está pressionado agora (em qualquer lugar da tela)?"""
        if sys.platform == "win32":
            try:
                import ctypes
                gas = ctypes.windll.user32.GetAsyncKeyState
                return any(gas(vk) & 0x8000 for vk in (0x01, 0x02, 0x04))
            except Exception:
                pass
        return QApplication.mouseButtons() != Qt.MouseButton.NoButton

    def _grab_hitimg(self):
        """Tira uma 'foto' da Lilith para saber quais pixels são personagem (têm opacidade)."""
        try:
            if not self.web.isVisible():
                return
            pm = self.web.grab()
            if pm.isNull():
                return
            w, h = self.web.width(), self.web.height()
            img = pm.toImage().scaled(
                w, h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.FastTransformation,
            )

            # Amostra grossa: que fração da área é "personagem"?
            total = opaque = 0
            for yy in range(0, h, 10):
                for xx in range(0, w, 10):
                    total += 1
                    if img.pixelColor(xx, yy).alpha() >= self.HIT_ALPHA:
                        opaque += 1
            ratio = opaque / total if total else 0

            # Captura sem transparência útil (vazia ou toda opaca): tudo fica clicável
            if ratio < 0.005 or ratio > 0.97:
                self._hit_fail += 1
                if self._hit_fail == 10 and not self._hit_diag_done:
                    self._hit_diag_done = True
                    print(f"[Lilith] Área clicável desativada: {ratio:.0%} da captura parece "
                          "'personagem' (a captura não trouxe transparência).")
                self._hit_img = None
                return
            self._hit_fail = 0
            self._hit_img = img
        except Exception as e:  # nunca derruba o programa por causa disto
            print(f"[Lilith] Erro ao capturar a silhueta: {e}")

    def _poll_cursor(self):
        try:
            # Com algum botão pressionado (arrastando/clicando) não muda nada
            if self._button_down():
                return
            pos = QCursor.pos()
            local = self.web.mapFromGlobal(pos)

            # Só atualiza a silhueta com o mouse por perto (evita capturas à toa)
            perto = QRect(-25, -25, self.web.width() + 50, self.web.height() + 50).contains(local)
            if perto:
                agora = time.monotonic()
                if (not self._cursor_near) or (agora - self._last_grab) >= self.GRAB_MS / 1000:
                    self._last_grab = agora
                    self._grab_hitimg()
            self._cursor_near = perto

            sobre_balao = (self.bubble.isVisible()
                           and self.bubble.geometry().contains(self.mapFromGlobal(pos)))
            sobre_lilith = self.web.rect().contains(local) and self._hit(local)
            self._set_click_through(not (sobre_balao or sobre_lilith))
        except Exception as e:
            print(f"[Lilith] Erro ao vigiar o cursor: {e}")

    def _hit(self, pos) -> bool:
        """True se o ponto (coordenadas do navegador embutido) está sobre a personagem."""
        img = self._hit_img
        if img is None:
            return True  # sem silhueta válida: comportamento antigo (tudo clicável)
        cx, cy = int(pos.x()), int(pos.y())
        r = self.HIT_RADIUS
        for dy in (-r, 0, r):
            for dx in (-r, 0, r):
                x, y = cx + dx, cy + dy
                if 0 <= x < img.width() and 0 <= y < img.height():
                    if img.pixelColor(x, y).alpha() >= self.HIT_ALPHA:
                        return True
        return False

    # ------------------------------------------------------------------
    # Mouse: o QWebEngineView "engole" os cliques, então repassamos para a janela
    # ------------------------------------------------------------------
    def _on_web_loaded(self, ok: bool):
        if not ok:
            print("[Lilith] Falha ao carregar renderer/index.html")
            return
        proxy = self.web.focusProxy()
        if proxy is not None:
            proxy.installEventFilter(self)
        if self.CLICK_THROUGH:
            self.poll_timer.start(self.POLL_MS)
        # Dá um tempo para o Spine carregar os 31 atlas e então lista animações e skins
        QTimer.singleShot(3000, self._load_catalog)

    def eventFilter(self, obj, event):
        t = event.type()
        if t in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            # Clique em área transparente: ignora (não arrasta, não abre, não fecha)
            if not self._hit(event.position()):
                return True
            if t == QEvent.Type.MouseButtonPress:
                self.mousePressEvent(event)
            else:
                self.mouseDoubleClickEvent(event)
            return True
        if t == QEvent.Type.MouseMove:
            self.mouseMoveEvent(event)
            return True
        return super().eventFilter(obj, event)

    def _welcome_message(self):
        self.bubble.start_speech()
        self.bubble.append_chunk("Estou conectada! Dê um duplo clique em mim para conversarmos.")
        self.bubble.finish_speech(7000)
        self.animator.play_emote("greeting")

    def position_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = screen.width() - self.width() - 80
        y = screen.height() - self.height() - 60
        self.move(x, y)

    def mousePressEvent(self, event):
        self._last_interaction = time.monotonic()
        if event.button() == Qt.MouseButton.RightButton:
            QApplication.quit()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.open_history()   # botão do meio (roda do mouse) abre o histórico
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self._last_interaction = time.monotonic()
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_interaction = time.monotonic()
            self.walker.stop_walking()
            self._dialog_open = True
            try:
                text, ok = QInputDialog.getText(
                    self, "Conversar com a Lilith", "O que você deseja dizer para a Lilith?"
                )
            finally:
                self._dialog_open = False
            if ok and text.strip():
                self.ask_lilith(text.strip())

    # ------------------------------------------------------------------
    # IA
    # ------------------------------------------------------------------
    def ask_lilith(self, prompt: str):
        self.bubble.start_speech()
        self.animator.play_state("thinking")
        self._first_chunk = True

        self.ai_thread = AIWorker(self.brain, prompt)
        self.ai_thread.chunk_signal.connect(self._on_chunk)
        self.ai_thread.animation_signal.connect(self._on_ai_animation)
        self.ai_thread.outfit_signal.connect(self.apply_preset)
        self.ai_thread.finished_signal.connect(self._on_finished)
        self.ai_thread.start()

    def _on_chunk(self, chunk: str):
        if self._first_chunk:
            self._first_chunk = False
            self.animator.start_talking()
        self.bubble.append_chunk(chunk)

    def _on_ai_animation(self, emote: str):
        self.animator.play_emote(emote)

    def _on_finished(self):
        # Mensagens curtas somem em 10 s como antes; as longas ficam mais tempo
        # (e de qualquer jeito ficam salvas no histórico).
        duracao = min(60000, max(10000, 3000 + 70 * len(self.bubble.current_text)))
        self.bubble.finish_speech(duracao)
        self.animator.stop_talking()
        QTimer.singleShot(3000, self.animator.reset_face)
        if self.history_win.isVisible():
            self.history_win.refresh()


if __name__ == "__main__":
    # Evita tela preta/transparência quebrada em algumas placas de vídeo:
    # descomente a linha abaixo se a Lilith aparecer com fundo preto.
    # os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # fechar o histórico não encerra a Lilith
    window = LilithOverlay(scale_factor=0.5)
    # Ao fechar: esconde o ícone da bandeja e atualiza o resumo (espera até 10 s)
    app.aboutToQuit.connect(window.shutdown)
    window.show()
    sys.exit(app.exec())
