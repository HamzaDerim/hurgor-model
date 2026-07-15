from scripts.vision import HurgorVision

def test_sistemi():
    print("🚀 HürGör Vision Motoru Başlatılıyor...")
    vision = HurgorVision()
    
    print("📡 İHA GPS/Telemetri verisi yükleniyor...")
    vision.load_telemetry("THYZ_2026_Ornek_Veri_1_translation.csv")
    
    # SENARYO: İHA uçarken 15. karede ekranın tam ortasında (u=960, v=540) bir nesne buldu.
    # Nesnenin kameraya olan uzaklığını tahmini 25 metre (z_depth) alıyoruz.
    u, v = 960, 540
    frame_id = 15 
    
    print(f"🎯 Nesne tespit edildi! Kameradaki Pikseli: (u={u}, v={v})")
    
    gercek_koordinat = vision.get_3d_coordinates(u, v, frame_id, z_depth_estimate=25.0)
    
    print("\n--- HÜRGÖR 3B HESAPLAMA SONUCU ---")
    print(f"🌍 Dünyadaki Gerçek Metrik Konumu:")
    print(f"X: {gercek_koordinat[0]:.2f} metre")
    print(f"Y: {gercek_koordinat[1]:.2f} metre")
    print(f"Z: {gercek_koordinat[2]:.2f} metre")

if __name__ == "__main__":
    test_sistemi()
