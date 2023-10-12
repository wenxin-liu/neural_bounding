import matplotlib.pyplot as plt


class Renderer2D:
    @staticmethod
    def render_comparison(target, prediction, original):
        fig, axs = plt.subplots(1, 3, figsize=(13, 5))

        axs[0].imshow(target, cmap='gray', vmin=0, vmax=1, interpolation='none')
        axs[1].imshow(prediction, cmap='gray', vmin=0, vmax=1, alpha=1.,
                      interpolation='none')
        axs[2].imshow(original, cmap='jet', vmin=0, vmax=1, interpolation='none')
        axs[2].imshow(prediction, cmap='gray', vmin=0, vmax=1, alpha=0.5,
                      interpolation='none')

        for ax in axs:
            ax.set_axis_off()

        plt.tight_layout()
        plt.show()
