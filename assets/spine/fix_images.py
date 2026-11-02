import os
from PIL import Image

# Resoluções exatas do arquivo .atlas
ATLAS_SIZES = {
    "Lilith.png": (4096, 4096),
    "Lilith_2.png": (4096, 4096),
    "Lilith_3.png": (4096, 4096),
    "Lilith_4.png": (4096, 4096),
    "Lilith_5.png": (4096, 4096),
    "Lilith_6.png": (4096, 4096),
    "Lilith_7.png": (4096, 4096),
    "Lilith_8.png": (4096, 4096),
    "Lilith_9.png": (4096, 4096),
    "Lilith_10.png": (4096, 4096),
    "Lilith_11.png": (4096, 4096),
    "Lilith_12.png": (4092, 4092),
    "Lilith_13.png": (4096, 4096),
    "Lilith_14.png": (4096, 4096),
    "Lilith_15.png": (4096, 4096),
    "Lilith_16.png": (4096, 4096),
    "Lilith_17.png": (4096, 4096),
    "Lilith_18.png": (4095, 4095),
    "Lilith_19.png": (4096, 4096),
    "Lilith_20.png": (4096, 4096),
    "Lilith_21.png": (4096, 4096),
    "Lilith_22.png": (4096, 4096),
    "Lilith_23.png": (4096, 4096),
    "Lilith_24.png": (4089, 4089),
    "Lilith_25.png": (4096, 4096),
    "Lilith_26.png": (4096, 4096),
    "Lilith_27.png": (4094, 4094),
    "Lilith_28.png": (4096, 4096),
    "Lilith_29.png": (4096, 4096),
    "Lilith_30.png": (4096, 4096),
    "Lilith_31.png": (3827, 3827),
}

def find_file_insensitive(target_png, search_dirs):
    target_base = os.path.splitext(target_png)[0].lower()
    valid_exts = [".png", ".jpg", ".jpeg"]
    
    for s_dir in search_dirs:
        for root, _, files in os.walk(s_dir):
            for f in files:
                fname, fext = os.path.splitext(f)
                if fname.lower() == target_base and fext.lower() in valid_exts:
                    return os.path.join(root, f)
    return None

def process_images():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    search_dirs = list(set([script_dir, cwd]))
    
    print(f"Buscando imagens nas pastas do projeto...")
    
    found_any = False
    for target_png, target_size in ATLAS_SIZES.items():
        found_path = find_file_insensitive(target_png, search_dirs)
        
        if not found_path:
            print(f"[X] Arquivo não encontrado: {target_png}")
            continue
        
        found_any = True
        dest_dir = os.path.dirname(found_path)
        output_path = os.path.join(dest_dir, target_png)
        
        try:
            with Image.open(found_path) as img:
                img = img.convert("RGBA")
                if img.size != target_size or found_path != output_path:
                    print(f"[!] Ajustando {found_path} -> {output_path} ({target_size})...")
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    resized_img.save(output_path, "PNG")
                    print(f"[V] {target_png} processado com sucesso!")
                else:
                    print(f"[V] {target_png} já está correto.")
        except Exception as e:
            print(f"[X] Erro ao processar {found_path}: {e}")
            
    if not found_any:
        print("\n[Aviso] Nenhuma imagem foi encontrada. Certifique-se de que as imagens (.png ou .jpg) estão dentro da pasta 'Assets/Resources/spinemodel/'.")

if __name__ == "__main__":
    process_images()