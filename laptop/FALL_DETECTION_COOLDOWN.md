# Fall Detection Cooldown Feature

## Overview
Fitur cooldown telah ditambahkan untuk mencegah deteksi jatuh berulang dalam periode waktu singkat. Hal ini membuat sistem lebih realistis dan mengurangi false positive yang berurutan.

## How It Works
- Ketika sistem mendeteksi jatuh, akan ada periode "cooldown" dimana deteksi jatuh berikutnya akan diabaikan
- Default cooldown period adalah 10 detik
- Selama periode cooldown, sistem tetap melakukan klasifikasi tetapi tidak akan memicu alert jatuh
- Log akan menunjukkan bahwa deteksi jatuh ditekan karena cooldown aktif

## Configuration

### Environment Variable
Tambahkan ke file `.env`:
```
FALL_COOLDOWN_SECONDS=10
```

### Command Line Argument
```bash
python server.py --cooldown 15
```

### Default Values
- Default cooldown: 10 seconds
- Minimum recommended: 5 seconds
- Maximum recommended: 30 seconds

## Benefits
1. **Realistic Detection**: Mencegah spam deteksi jatuh dalam satu kejadian
2. **Better User Experience**: Mengurangi notifikasi berulang yang tidak perlu
3. **Improved Accuracy**: Fokus pada kejadian jatuh yang berbeda, bukan frame berurutan
4. **Configurable**: Dapat disesuaikan berdasarkan kebutuhan aplikasi

## Example Log Output
```
2024-01-15 10:30:15 - INFO - Classification: Jatuh (confidence: 0.85) - FALL ALERT!
2024-01-15 10:30:16 - INFO - Fall detection suppressed - cooldown active (1.2s < 10s)
2024-01-15 10:30:17 - INFO - Fall detection suppressed - cooldown active (2.1s < 10s)
2024-01-15 10:30:26 - INFO - Classification: Jatuh (confidence: 0.78) - FALL ALERT!
```

## Technical Implementation
- Menggunakan `datetime.now()` untuk tracking waktu deteksi terakhir
- Cooldown dihitung dalam detik dengan presisi desimal
- Thread-safe implementation untuk multiple concurrent requests
- Tidak mempengaruhi logging normal activity atau deteksi di bawah threshold

## Troubleshooting
- Jika cooldown terlalu pendek: Masih ada deteksi berulang
- Jika cooldown terlalu panjang: Kejadian jatuh yang berbeda mungkin terlewat
- Recommended: Mulai dengan 10 detik dan sesuaikan berdasarkan testing