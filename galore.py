""" GaLore
https://arxiv.org/abs/2403.03507

Copyright 2025 NoteDance
"""
import tensorflow as tf
from keras.src.optimizers import optimizer


class GaLore(optimizer.Optimizer):
    def __init__(
        self,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-6,
        weight_decay=0.0,
        rank=None,
        update_proj_gap=None,
        scale=None,
        projection_type=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        loss_scale_factor=None,
        gradient_accumulation_steps=None,
        name="galore",
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type
    
    def reset(self):
        self._iterations.assign(0)
        for var in self._trainable_variables:
            self.exp_avg[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg"
                                                    )
            self.exp_avg_sq[self._get_variable_index(var)] =  self.add_variable_from_reference(
                                                        reference_variable=var, name="exp_avg_sq"
                                                    )

    def build(self, var_list):
        if self.built:
            return
        super().build(var_list)
        self.exp_avg = []
        self.exp_avg_sq = []
        self.projector = []
        for var in var_list:
            self.exp_avg.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg"
                                                    ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                                reference_variable=var, name="exp_avg_sq"
                                                    ))
            self.projector.append(None)

    def update_step(self, gradient, variable, learning_rate):
        lr = tf.cast(learning_rate, variable.dtype)
                
        step = tf.cast(self.iterations + 1, variable.dtype)
        
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2_sq = tf.sqrt(1 - self.beta2 ** step)
        
        step_size = lr * bias_correction2_sq / bias_correction1
        
        if tf.keras.backend.is_sparse(gradient):
            raise RuntimeError(
                'GaLore does not support sparse gradients')
        
        if self.rank is not None and len(variable.shape) > 1:
            if self.projector[self._get_variable_index(variable)] is None:
                self.projector[self._get_variable_index(variable)] = GaLoreProjector(
                    rank=self.rank,
                    update_proj_gap=self.update_proj_gap,
                    scale=self.scale,
                    projection_type=self.projection_type,
                )

            grad = self.projector[self._get_variable_index(variable)].project(gradient, step)
        
        if self.weight_decouple:
            variable.assign(variable * (1.0 - self.weight_decay * (1.0 if self.fixed_decay else lr)))
        
        exp_avg = self.exp_avg[self._get_variable_index(variable)]
        exp_avg_sq = self.exp_avg_sq[self._get_variable_index(variable)]
        exp_avg.assign(exp_avg * self.beta1 + gradient * (1.0 - self.beta1))
        exp_avg_sq.assign(exp_avg_sq * self.beta2 + grad * grad * (1.0 - self.beta2))
        
        de_nom = tf.sqrt(exp_avg_sq) + self.epsilon
        
        norm_grad = exp_avg / de_nom
    
        if self.rank is not None and len(variable.shape) > 1:
            norm_grad = self.projector[self._get_variable_index(variable)].project_back(norm_grad)
        
        variable.assign_add(norm_grad * -step_size)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "rank": self.rank,
                "update_proj_gap": self.update_proj_gap,
                "scale": self.scale,
                "projection_type": self.projection_type,
                "projector": self.projector,
            }
        )
        return config
	
    def _apply_weight_decay(self, variables):
        pass


PROJECTION_TYPE = Literal['std', 'reverse_std', 'right', 'left', 'full', 'random']


class GaLoreProjector:
    def __init__(
        self,
        rank: int = 128,
        update_proj_gap: int = 50,
        scale: float = 1.0,
        projection_type: PROJECTION_TYPE = 'std',
        **kwargs,
    ):
        self.rank = rank
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.projection_type = projection_type

        self.ortho_matrix = None
    
    @staticmethod
    def get_orthogonal_matrix(
        weights, rank: int, projection_type: str, from_random_matrix: bool = False
    ):
        if projection_type not in {'right', 'left', 'full'}:
            raise ValueError('projection_type should be one of left, right or full')
        
        original_type = weights.dtype
        is_float: bool = original_type == tf.float32

        if not from_random_matrix:
            u, s, v = tf.linalg.svd(weights if is_float else tf.cast(weights, dtype=tf.float32), full_matrices=False)
            vh = tf.transpose(v)
        else:
            m, n = weights.shape[0], weights.shape[1]
            u = tf.random.normal((m, rank), dtype=original_type) / math.sqrt(rank)
            vh = tf.random.normal((rank, n), dtype=original_type) / math.sqrt(rank)

        if projection_type == 'right':
            b = vh[:rank, :]
            return b
        if projection_type == 'left':
            a = u[:, :rank]
            return a if is_float else tf.cast(a, dtype=original_type)

        a = u[:, :rank]
        b = vh[:rank, :]
        return ((a, b) if is_float else (tf.cast(a, dtype=original_type), tf.cast(b, dtype=original_type)))

    def get_low_rank_grad_std(self, grad, steps, from_random_matrix: bool):
        if grad.shape[0] >= grad.shape[1]:
            def true_fn():
                self.ortho_matrix = self.get_orthogonal_matrix(
                    grad, self.rank, projection_type='right', from_random_matrix=from_random_matrix
                )
            
            def false_fn():
                pass
            
            tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
            return tf.matmul(grad, tf.transpose(self.ortho_matrix))
        
        def true_fn():
            self.ortho_matrix = self.get_orthogonal_matrix(
                grad, self.rank, projection_type='left', from_random_matrix=from_random_matrix
            )
        
        def false_fn():
            pass
        
        tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
        return tf.matmul(tf.transpose(self.ortho_matrix), grad)

    def get_low_rank_grad_reverse_std(self, grad, steps: int, from_random_matrix: bool):
        if grad.shape[0] >= grad.shape[1]:
            def true_fn():
                self.ortho_matrix = self.get_orthogonal_matrix(
                    grad, self.rank, projection_type='left', from_random_matrix=from_random_matrix
                )
            
            def false_fn():
                pass
            
            tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
            return tf.matmul(tf.transpose(self.ortho_matrix), grad)
        
        def true_fn():
            self.ortho_matrix = self.get_orthogonal_matrix(
                grad, self.rank, projection_type='right', from_random_matrix=from_random_matrix
            )
        
        def false_fn():
            pass
        
        tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
        return tf.matmul(grad, tf.transpose(self.ortho_matrix))

    def get_low_rank_grad_right(self, grad, steps: int, from_random_matrix: bool):
        def true_fn():
            self.ortho_matrix = self.get_orthogonal_matrix(
                grad, self.rank, projection_type='right', from_random_matrix=from_random_matrix
            )
        
        def false_fn():
            pass
        
        tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
        return tf.matmul(grad, tf.transpose(self.ortho_matrix))

    def get_low_rank_grad_left(self, grad, steps: int, from_random_matrix: bool):
        def true_fn():
            self.ortho_matrix = self.get_orthogonal_matrix(
                grad, self.rank, projection_type='left', from_random_matrix=from_random_matrix
            )
        
        def false_fn():
            pass
        
        tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
        return tf.matmul(tf.transpose(self.ortho_matrix), grad)

    def get_low_rank_grad_full(self, grad, steps: int, from_random_matrix: bool):
        def true_fn():
            self.ortho_matrix = self.get_orthogonal_matrix(
                grad, self.rank, projection_type='full', from_random_matrix=from_random_matrix
            )
        
        def false_fn():
            pass
        
        tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
        a, b = self.ortho_matrix
        return tf.matmul(tf.matmul(tf.transpose(a), grad), tf.transpose(b))

    def get_low_rank_grad_random(self, grad, steps: int, from_random_matrix: bool):
        is_right = grad.shape[0] >= grad.shape[1]
        proj_type = 'right' if is_right else 'left'
        def true_fn():
            self.ortho_matrix = self.get_orthogonal_matrix(
                grad, self.rank, projection_type=proj_type, from_random_matrix=from_random_matrix
            )
        
        def false_fn():
            pass
        
        tf.cond(tf.logical_or(self.ortho_matrix is None, steps % self.update_proj_gap == 0), true_fn, false_fn)
        if is_right:
            return tf.matmul(grad, tf.transpose(self.ortho_matrix))
        else:
            return tf.matmul(tf.transpose(self.ortho_matrix), grad)

    def project(self, full_rank_grad, steps, from_random_matrix: bool = False):
        if self.projection_type == 'std':
            return self.get_low_rank_grad_std(full_rank_grad, steps, from_random_matrix)
        elif self.projection_type == 'reverse_std':
            return self.get_low_rank_grad_reverse_std(full_rank_grad, steps, from_random_matrix)
        elif self.projection_type == 'right':
            return self.get_low_rank_grad_right(full_rank_grad, steps, from_random_matrix)
        elif self.projection_type == 'left':
            return self.get_low_rank_grad_left(full_rank_grad, steps, from_random_matrix)
        elif self.projection_type == 'full':
            return self.get_low_rank_grad_full(full_rank_grad, steps, from_random_matrix)
        elif self.projection_type == 'random':
            return self.get_low_rank_grad_random(full_rank_grad, steps, from_random_matrix)
        else:
            raise NotImplementedError

    def project_back(self, low_rank_grad):
        if self.projection_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                return tf.matmul(low_rank_grad, self.ortho_matrix) * self.scale
            else:
                return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        elif self.projection_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                return tf.matmul(self.ortho_matrix, tf.transpose(low_rank_grad)) * self.scale
            else:
                return tf.matmul(low_rank_grad, tf.transpose(self.ortho_matrix)) * self.scale
        elif self.projection_type == 'right':
            return tf.matmul(low_rank_grad, tf.transpose(self.ortho_matrix)) * self.scale
        elif self.projection_type == 'left':
            return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        elif self.projection_type == 'full':
            a, b = self.ortho_matrix
            return tf.matmul(tf.matmul(a, low_rank_grad), tf.transpose(b)) * self.scale
        elif self.projection_type == 'random':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                return tf.matmul(low_rank_grad, tf.transpose(self.ortho_matrix)) * self.scale
            else:
                return tf.matmul(self.ortho_matrix, low_rank_grad) * self.scale
        else:
            raise NotImplementedError
