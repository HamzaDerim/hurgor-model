import pandas as pd
import matplotlib.pyplot as plt
import os
import math

def ciz():
    # Ana proje dizinini bul
    proje_dizini = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_yolu = os.path.join(proje_dizini, "hurgor_koordinat_raporu.csv")
    
    print("🗺️ Gelişmiş uçuş haritası oluşturuluyor...")
    df = pd.read_csv(csv_yolu)
    
    # İHA'nın rotasını frame sırasına göre al
    rota = df.drop_duplicates(subset=['Frame_ID'])[['UAV_Global_X', 'UAV_Global_Y']]
    
    x = rota['UAV_Global_X'].values
    y = rota['UAV_Global_Y'].values
    
    plt.figure(figsize=(12, 9))
    
    # Rotayı çiz
    plt.plot(x, y, label='Uçuş İzi', color='royalblue', linewidth=2, zorder=1, alpha=0.6)
    
    # Başlangıç Noktası (Orijin)
    plt.scatter(x[0], y[0], color='green', marker='o', s=150, label='Kalkış Noktası (0,0)', zorder=2)
    
    # Gidiş yönünü gösteren oklar ve MESAFE METİNLERİ
    adim_sayisi = max(1, len(x) // 15) # Ok sayısını dengeler
    for i in range(0, len(x)-1, adim_sayisi):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        mesafe = math.hypot(dx, dy)
        
        # Sadece belirgin hareketlerde ok ve yazı ekle
        if mesafe > 0.5: 
            plt.arrow(x[i], y[i], dx, dy, shape='full', lw=0, length_includes_head=True, head_width=2.0, color='darkblue', zorder=3)
            # Okun orta noktasına kaç metre olduğunu yaz
            mid_x = x[i] + (dx / 2)
            mid_y = y[i] + (dy / 2)
            plt.text(mid_x, mid_y + 1, f"{mesafe:.1f}m", fontsize=9, color='darkred', fontweight='bold', ha='center')
    
    # İHA SİMGESİ (Son Konum - Üçgen Marker)
    # dx ve dy'ye göre İHA'nın burnunun nereye baktığını hesaplayabiliriz, ama şimdilik belirgin bir üçgen koyuyoruz
    plt.scatter(x[-1], y[-1], color='red', marker='^', s=400, label='İHA (Anlık Konum)', zorder=4)
    
    # Harita Görsel Ayarları
    plt.title('HürGör Dinamik Görsel Odometri (GPS-Denied)', fontsize=16, fontweight='bold')
    plt.xlabel('X Ekseni (Metre)', fontsize=12)
    plt.ylabel('Y Ekseni (Metre)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend(loc='upper left', fontsize=11)
    
    # Arka planı hafif gri yapalım ki renkler patlasın
    plt.gca().set_facecolor('#f4f4f4')
    
    # Dosyayı kaydet ve ekranda göster
    plt.tight_layout()
    kayit_yolu = os.path.join(proje_dizini, "hurgor_ucus_haritasi.png")
    plt.savefig(kayit_yolu, dpi=300)
    print(f"✅ Harita başarıyla kaydedildi: {kayit_yolu}")
    plt.show()

if __name__ == '__main__':
    ciz()