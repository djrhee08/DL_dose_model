import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, MaxPooling3D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, Input

###############################################################################
# Utility Blocks
###############################################################################
def conv_block_3d(x, n_filters, kernel_size=(3,3,3), activation='relu', padding='same'):
    x = Conv3D(n_filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    
    x = Conv3D(n_filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    return x

def encoder_block_3d(x, n_filters, pool_size=(2,2,2)):
    f = conv_block_3d(x, n_filters=n_filters)
    p = MaxPooling3D(pool_size=pool_size)(f)
    return f, p

def decoder_block_3d(x, skip_features, n_filters, kernel_size=(2,2,2), strides=(2,2,2)):
    x = Conv3DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = concatenate([x, skip_features])
    x = conv_block_3d(x, n_filters)
    return x

###############################################################################
# 1) Coarse 3D U-Net that returns final segmentation + final decoder feature
###############################################################################
def create_coarse_3d_unet(
    input_shape=(128, 128, 128, 3),
    n_filters_base=16, 
    n_classes=1, 
    final_activation='sigmoid'
):
    """
    A smaller/lower-resolution 3D U-Net for coarse segmentation.
    Returns both:
      - coarse_seg: the final segmentation
      - d1: the final feature map in the decoder
    """
    inputs = Input(input_shape)
    
    # Encoder
    f1, p1 = encoder_block_3d(inputs, n_filters_base)      # level 1
    f2, p2 = encoder_block_3d(p1, n_filters_base * 2)       # level 2
    f3, p3 = encoder_block_3d(p2, n_filters_base * 4)       # level 3

    # Bottleneck
    bn = conv_block_3d(p3, n_filters_base * 8)
    
    # Decoder
    d3 = decoder_block_3d(bn, f3, n_filters_base * 4)       # level 3
    d2 = decoder_block_3d(d3, f2, n_filters_base * 2)       # level 2
    d1 = decoder_block_3d(d2, f1, n_filters_base)           # level 1
    
    # Final segmentation output
    coarse_seg = Conv3D(n_classes, (1,1,1), padding='same', activation=final_activation)(d1)
    
    # We return both the coarse segmentation and the final decoder feature map d1
    model = Model(inputs, [coarse_seg, d1], name='Coarse3DUNet')
    return model


def create_fine_3d_unet(
    input_shape=(128, 128, 128, 3+16),
    n_filters_base=32, 
    n_classes=1, 
    final_activation='sigmoid'
):
    """
    Fine 3D U-Net for refined segmentation on ROI or bounding box,
    or on a multi-channel input (original ROI + coarse features).
    """
    inputs = Input(input_shape)  # could be (N, 128, 128, 128, C)
    
    # Encoder
    f1, p1 = encoder_block_3d(inputs, n_filters_base)
    f2, p2 = encoder_block_3d(p1, n_filters_base*2)
    f3, p3 = encoder_block_3d(p2, n_filters_base*4)
    f4, p4 = encoder_block_3d(p3, n_filters_base*8)
    
    # Bottleneck
    bn = conv_block_3d(p4, n_filters_base*16)
    
    # Decoder
    d4 = decoder_block_3d(bn, f4, n_filters_base*8)
    d3 = decoder_block_3d(d4, f3, n_filters_base*4)
    d2 = decoder_block_3d(d3, f2, n_filters_base*2)
    d1 = decoder_block_3d(d2, f1, n_filters_base)
    
    # Output
    fine_seg = Conv3D(n_classes, (1,1,1), padding='same', activation=final_activation)(d1)
    
    model = Model(inputs, fine_seg, name='Fine3DUNet')
    return model

def create_cascade_3d_unet():
        
    """
    Demonstrates conceptually how to build a multi-input cascade in one Keras model.
    In real practice, you'd do offline or custom-layers for ROI extraction & resizing.
    """
    # 1) Instantiate models
    coarse_model = create_coarse_3d_unet(input_shape=(128, 128, 128, 3))  # returns [coarse_seg, d1]
    fine_model   = create_fine_3d_unet(input_shape=(128, 128, 128, 3 + 16)) 
    
    # 2) Inputs
    coarse_inputs = Input((128, 128, 128, 3), name='coarse_inputs')
    highres_inputs = Input((128, 128, 128, 3), name='highres_inputs')
    
    # 3) Forward pass in coarse model
    coarse_seg, coarse_features = coarse_model(coarse_inputs)
    
    # 4) Concatenate ROI inputs for fine net
    # In practice, you'd do something like: 
    #   fine_inputs = layers.Concatenate(axis=-1)([roi_highres, roi_features])
    #
    # For demonstration, let's pretend highres_inputs is the ROI and that coarse_features
    # was magically resized. We'll call a dummy "Upsampling3D" on coarse_features just
    # as a placeholder, to get it to 128^3:
    
    # shape might become (None, 128, 128, 128, 3 + 16) if the factor is correct for your data
    fine_inputs = layers.Concatenate(axis=-1)([highres_inputs, coarse_features])
    
    # 5) Fine segmentation
    fine_seg = fine_model(fine_inputs)
    
    # 6) Build a multi-input, multi-output cascade model
    cascade_model = Model(
        inputs=[coarse_inputs, highres_inputs],
        outputs=[coarse_seg, fine_seg],
        name='Cascade3DUNet'
    )
    
    return cascade_model


###############################################################################
# Example usage
###############################################################################
if __name__ == "__main__":
    # Build the cascade
    cascade_net = create_cascade_3d_unet()
    cascade_net.summary()
    
    # Dummy data
    import numpy as np
    x_coarse = np.random.rand(1, 128, 128, 128, 3).astype(np.float32)
    x_fine   = np.random.rand(1, 128, 128, 128, 3).astype(np.float32)
    
    # Forward pass
    out_coarse_seg, out_fine_seg = cascade_net.predict([x_coarse, x_fine])
    print("Out coarse seg shape:", out_coarse_seg.shape)
    print("Out fine seg shape:", out_fine_seg.shape)