"""
端到端测试：测试完整的工作流程
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestEndToEnd:
    """端到端功能测试"""
    
    @pytest.fixture
    def sample_config(self):
        """创建示例配置"""
        from molmo.models.config_molmoe import MolmoConfig
        
        return MolmoConfig(
            d_model=128,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            vocab_size=1000,
            max_sequence_length=256,
            init_device="cpu",
            vision_backbone=None,
            layer_norm_eps=1e-5,
        )
    
    def test_config_to_model_to_forward(self, sample_config):
        """测试从配置到模型到前向传播的完整流程"""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 1. 从配置创建模型
        model = MolmoModel(sample_config)
        model.eval()
        
        # 2. 创建输入
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        # 3. 前向传播
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        # 4. 验证输出
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, sample_config.d_model)
        print("✓ End-to-end: Config -> Model -> Forward works")
    
    def test_training_workflow(self, sample_config):
        """测试完整的训练工作流程"""
        from molmo.models.modeling_molmoe import MolmoModel
        from molmo.optim import build_optimizer, build_scheduler
        
        # 1. 创建模型
        model = MolmoModel(sample_config)
        model.train()
        
        # 2. 创建优化器和调度器
        optimizer = build_optimizer(
            model,
            optimizer_type="adamw",
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        scheduler = build_scheduler(
            optimizer,
            scheduler_type="cosine_with_warmup",
            num_training_steps=100,
            num_warmup_steps=10,
        )
        
        # 3. 模拟多个训练步骤
        for step in range(3):
            batch_size = 2
            seq_len = 10
            input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids)
            loss = outputs.last_hidden_states.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print("✓ End-to-end training workflow works")
    
    def test_model_state_management(self, sample_config):
        """测试模型状态管理"""
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 1. 创建模型
        model1 = MolmoModel(sample_config)
        model1.train()
        
        # 2. 获取状态字典
        state_dict = model1.state_dict()
        
        # 3. 创建新模型并加载状态
        model2 = MolmoModel(sample_config)
        model2.load_state_dict(state_dict)
        
        # 4. 验证两个模型输出相同
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, sample_config.vocab_size, (batch_size, seq_len))
        
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            out1 = model1(input_ids=input_ids)
            out2 = model2(input_ids=input_ids)
        
        # 检查输出是否相同（允许小的数值误差）
        assert torch.allclose(out1.last_hidden_states, out2.last_hidden_states, atol=1e-6)
        print("✓ Model state management works")
    
    def test_config_conversion_workflow(self):
        """测试配置转换的完整工作流程"""
        from molmo.config import ModelConfig, model_config_to_molmo_config
        from molmo.models.modeling_molmoe import MolmoModel
        
        # 1. 创建训练配置
        train_cfg = ModelConfig(
            d_model=128,
            n_heads=2,
            n_layers=1,
            max_sequence_length=256,
            vocab_size=1000,
        )
        
        # 2. 转换为 HF 配置
        hf_cfg = model_config_to_molmo_config(train_cfg)
        
        # 3. 使用 HF 配置创建模型
        model = MolmoModel(hf_cfg)
        
        # 4. 验证模型可以运行
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, hf_cfg.vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
        
        assert outputs.last_hidden_states.shape == (batch_size, seq_len, hf_cfg.d_model)
        print("✓ Config conversion workflow works")



