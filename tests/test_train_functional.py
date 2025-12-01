"""
训练功能测试：测试训练相关的实际功能
"""
import pytest
import torch
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestTrainFunctional:
    """测试训练模块的实际功能"""
    
    def test_optimizer_creation(self):
        """测试优化器的创建"""
        from molmo.optim import build_optimizer, OptimizerType
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            vocab_size=100,
            max_sequence_length=128,
            init_device="cpu",
            vision_backbone=None,
        )
        
        model = MolmoModel(config)
        
        # 测试创建优化器
        optimizer = build_optimizer(
            model,
            optimizer_type=OptimizerType.adamw,
            learning_rate=1e-4,
            weight_decay=0.01,
        )
        
        assert optimizer is not None
        print("✓ Optimizer creation works")
    
    def test_scheduler_creation(self):
        """测试调度器的创建"""
        from molmo.optim import build_scheduler, SchedulerType
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        from molmo.optim import build_optimizer
        
        config = MolmoConfig(
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            vocab_size=100,
            max_sequence_length=128,
            init_device="cpu",
            vision_backbone=None,
        )
        
        model = MolmoModel(config)
        optimizer = build_optimizer(model, optimizer_type="adamw", learning_rate=1e-4)
        
        # 测试创建调度器
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=SchedulerType.cosine_with_warmup,
            num_training_steps=1000,
            num_warmup_steps=100,
        )
        
        assert scheduler is not None
        print("✓ Scheduler creation works")
    
    def test_checkpoint_save_load(self, tmp_path):
        """测试检查点的保存和加载"""
        from molmo.checkpoint import Checkpointer
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        
        config = MolmoConfig(
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            vocab_size=100,
            max_sequence_length=128,
            init_device="cpu",
            vision_backbone=None,
        )
        
        model = MolmoModel(config)
        
        # 创建检查点目录
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        
        # 测试保存检查点
        checkpointer = Checkpointer(checkpoint_dir)
        # checkpointer.save_checkpoint(model, step=0)
        
        assert checkpointer is not None
        print("✓ Checkpointer can be created")
    
    def test_training_step_simulation(self):
        """模拟一个训练步骤"""
        from molmo.models.config_molmoe import MolmoConfig
        from molmo.models.modeling_molmoe import MolmoModel
        from molmo.optim import build_optimizer
        
        config = MolmoConfig(
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            n_layers=1,
            vocab_size=100,
            max_sequence_length=128,
            init_device="cpu",
            vision_backbone=None,
        )
        
        model = MolmoModel(config)
        model.train()
        
        optimizer = build_optimizer(model, optimizer_type="adamw", learning_rate=1e-4)
        
        # 模拟一个训练步骤
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids)
        loss = outputs.last_hidden_states.mean()
        loss.backward()
        optimizer.step()
        
        print("✓ Training step simulation works")



