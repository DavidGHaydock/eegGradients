import numpy as np
from sklearn.decomposition import PCA


def compute_pca(connectivity_matrix: np.ndarray,
                n_components: int
                ) -> np.ndarray:
    """
    Compute PCA gradients from connectivity matrix.

    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        NxN connectivity matrix
    num_gradients : int
        Number of gradients (PCs) to return

    Returns:
    --------
    gradients
    """

    pca = PCA(n_components=2)
    pca.fit(connectivity_matrix)

    return pca.components_


def compute_diffusion_embedding(connectivity_matrix, num_gradients):
    """
    Compute diffusion embedding gradients from connectivity matrix.

    Parameters:
    -----------
    connectivity_matrix : numpy.ndarray
        NxN connectivity matrix
    num_gradients : int
        Number of gradients (eigenvectors) to return

    Returns:
    --------
    gradients
        gradients: NxK matrix of K gradients
    """
    # Normalize the connectivity matrix to get a Markov matrix
    row_sums = np.sum(connectivity_matrix, axis=1)
    markov_matrix = connectivity_matrix / row_sums[:, np.newaxis]

    # Compute the eigenvectors and eigenvalues of the Markov matrix
    eigenvalues, eigenvectors = np.linalg.eig(markov_matrix)

    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[idx]
    sorted_eigenvectors = eigenvectors[:, idx]

    # Return the top 'num_gradients' eigenvectors (gradients) and eigenvalues
    gradients = sorted_eigenvectors[:, :num_gradients]
    eigenvalues = sorted_eigenvalues[:num_gradients]

    return gradients


def compute_gradients(connectivity_matrix, num_gradients, decomposition_type=None):
    """
    Compute gradients for EEG connectivity and store in EEG object.

    Parameters:
    -----------
    EEG : dict-like object
        EEG data structure with connectivity matrix
    num_gradients : int
        Number of gradients to compute
    decomposition_type : str, optional
        Placeholder for potential future decomposition methods

    Returns:
    --------
    gradients : np.ndarray

    eigenvalues: np.ndarray
    """

    if decomposition_type == 'pca':
        gradients = compute_pca(
            connectivity_matrix,
            num_gradients
        )

    elif decomposition_type == 'diff_embed':
        # Compute diffusion embedding gradients
        gradients = compute_diffusion_embedding(
            connectivity_matrix,
            num_gradients
        )


    return
