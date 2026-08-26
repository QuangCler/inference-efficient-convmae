import torch
import torch.nn as nn
import torch.nn.functional as F

class GhostV2BlockMasked(nn.Module):
    """
    GhostNetV2 block với masked convolution cho MAE pretraining.
    """
    def __init__(self, dim, out_dim=None, kernel_h=9, kernel_w=9):
        super().__init__()
        out_dim = out_dim or dim
        mid_channels = out_dim // 2
        
        # --- Ghost Module Paths ---
        # Path 1: Primary 1x1 
        self.primary_conv = nn.Conv2d(dim, mid_channels, 1, bias=False)
        
        # Path 2: Cheap 3x3 DW 
        self.cheap_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, 
                                    padding=1, groups=mid_channels, bias=False)
        # bias = False vì độ lệch bias sẽ làm hỏng giá trị 0 của masked input, dẫn đến rò rỉ thông tin qua các convolution sau đó.

        # --- DFC Attention Paths (CẦN MASK) ---
        self.input_proj = nn.Conv2d(dim, out_dim, 1, bias=False)
        self.horizontal_fc = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_w), 
                                       padding=(0, kernel_w//2), groups=out_dim, bias=False)
        self.vertical_fc = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_h, 1), 
                                     padding=(kernel_h//2, 0), groups=out_dim, bias=False)
        
        # --- SE Module ---
        squeeze = max(out_dim // 4, 16)
        self.se_reduce = nn.Conv2d(out_dim, squeeze, 1, bias=False)
        self.se_expand = nn.Conv2d(squeeze, out_dim, 1, bias=False)
        # Squeeze-and-Excitation module giúp mô hình tự động đánh giá xem "kênh đặc trưng nào quan trọng hơn" để nhân trọng số lên.

        # Skip connection
        self.shortcut = nn.Conv2d(dim, out_dim, 1, bias=False) if dim != out_dim else nn.Identity() 
        # Skip connection cộng thẳng đầu vào vào đầu ra, giúp chống suy biến đạo hàm khi mạng sâu lên.


    def forward(self, x, mask=None):
        # Tại Inference, mask = None, chạy bình thường (Dense)
        if mask is None:
            y_primary = self.primary_conv(x)
            y_cheap = self.cheap_conv(y_primary)
            y_ghost = torch.cat([y_primary, y_cheap], dim=1)
            
            z = self.input_proj(x)
            a = torch.sigmoid(self.vertical_fc(self.horizontal_fc(z)))
            out = y_ghost * a
            
            se = out.mean(dim=[2, 3], keepdim=True)
            se = torch.sigmoid(self.se_expand(F.relu(self.se_reduce(se))))
            return (out * se) + self.shortcut(x)

        # Tại Pre-training
        # 1. Ghost Module
        y_primary = self.primary_conv(x)                # [B, C/2, H, W]
        y_cheap_in = y_primary * mask                   # Pre-masking
        y_cheap = self.cheap_conv(y_cheap_in) * mask    # Post-masking để dập rò rỉ
        y_ghost = torch.cat([y_primary, y_cheap], dim=1)
        
        # 2. DFC Attention
        z = self.input_proj(x * mask)                   # Mask input của DFC
        a_h = self.horizontal_fc(z)
        a_v = self.vertical_fc(a_h)
        attention = torch.sigmoid(a_v) * mask           # Mask attention map
        
        # 3. Combine
        out = y_ghost * attention
        
        # 4. SE Module (Không rò rỉ vì tính trung bình chỉ trên các vùng mask > 0)
        # Lưu ý: Tính trung bình chính xác cần chia cho tổng số pixel nhìn thấy, 
        # nhưng để đơn giản và ổn định ta có thể xấp xỉ bằng mean thông thường kết hợp mask
        se = out.mean(dim=[2, 3], keepdim=True)
        se = torch.sigmoid(self.se_expand(F.relu(self.se_reduce(se))))
        out = out * se
        
        # return out + self.shortcut(x)
        return out + self.shortcut(x * mask)