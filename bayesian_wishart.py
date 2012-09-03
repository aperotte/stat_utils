import numpy as np
import scipy.special as special

def wishart_ll(L, W, nu):
  """
  Formula for the wishart distribution based on the
  appendix in Bishop - corroborated by Wikipedia
  definition
  """
  nu = float(nu)
  ll = 0
  D = float(W.shape[0])
  # Evaluate B_W_nu
  B_W_nu = 0
  B_W_nu += -nu/2.*np.log(np.linalg.det(W))
  temp = nu*D/2.*np.log(2)
  temp += D*(D-1)/4.*np.log(np.pi)
  temp += np.sum(special.gammaln((nu+1-np.arange(1,D+1))/2.))
  B_W_nu += -temp
  # Continue with ll eval
  ll += B_W_nu
  ll += (nu-D-1)/2.*np.log(np.linalg.det(L))
  ll += -0.5*np.trace(np.dot(np.linalg.inv(W),L))
  return ll


def inv_wishart_ll(L, W, nu):
  """
  Based on modified wikipedia definition.  Takes a precision matrix
  and uses (nu-D-1) [m-p-1 on wikipedia] instead of (nu+D+1) [m+p+1]
  as defined on wikipedia.
  """
  nu = float(nu)
  ll = 0
  D = float(W.shape[0])
  # Evaluate B_W_nu
  B_W_nu = 0
  B_W_nu += nu/2.*np.log(np.linalg.det(W))
  temp = nu*D/2.*np.log(2)
  temp += D*(D-1)/4.*np.log(np.pi)
  temp += np.sum(special.gammaln((nu+1-np.arange(1,D+1))/2.))
  B_W_nu += -temp
  # Continue with ll eval
  ll += B_W_nu
  ll += -(nu+D+1)/2.*np.log(np.linalg.det(L))
  ll += -0.5*np.trace(np.dot(W,np.linalg.inv(L)))
  return ll


def mvn_ll(sample, mean, cov):
    D = mean.shape[0]
    norm = -D/2.*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov))
    sample_diff = sample-mean
    like = -0.5*np.sum(np.dot(sample_diff,np.linalg.inv(cov))*sample_diff,1)
    return norm + like


def mvn_ll_single(sample, mean, cov):
    D = mean.shape[0]
    norm = -D/2.*np.log(2*np.pi) - 0.5*np.log(np.linalg.det(cov))
    sample_diff = sample-mean
    like = -0.5*np.sum(np.dot(sample_diff,np.linalg.inv(cov))*sample_diff)
    return norm + like



def post_norm_wish(cluster_points, beta_0=None, nu_0=None, m_0=None, W_0=None):
    """
    Default parameterization - Not passed directly because
    first parameter must be inspected
    beta_0 = 1.
    nu_0 = dims
    m_0 = np.zeros(dims)
    W_0 = np.eye(dims)*1
    
    This is the posterior for a normal inverse-wishart.
    Given the multivariate normally
    distributed observations, this function returns
    W_p a covariance-like matrix parameter for the inverse-
    wishart.  Invert for wishart parameterization.
    """
    if len(cluster_points.shape) == 2:
        dims = cluster_points.shape[1]
    elif len(cluster_points.shape) == 1:
        dims = 1
    else:
        raise Exception('Array much be of one or two dimensions!')
    if not beta_0:
        beta_0 = 1
    if not nu_0:
        nu_0 = dims
    if not m_0:
        m_0 = np.zeros(dims)
    if not W_0:
        W_0 = np.eye(dims)
    N_0 = float(cluster_points.shape[0])
    big_cluster_points = cluster_points.sum(axis=0)
    x_bar = cluster_points.mean(axis=0)
    diff = cluster_points-x_bar
    if len(cluster_points) > 1:
        S = np.cov(cluster_points.T)*(N_0-1)
    else:
        S = np.outer(cluster_points,cluster_points)
    # covariance weight for mean
    beta_p = beta_0 + N_0
    # degress of freedom
    nu_p = nu_0 + N_0
    # mean
    m_p = (beta_0*m_0 + big_cluster_points)/(beta_0+N_0)
    # Covariance
    W_p = (W_0) + S + ((beta_0*N_0)/(beta_0+N_0))*np.outer(m_0-x_bar,m_0-x_bar)
    return beta_p, nu_p, m_p, W_p

def jll(cluster_points, mu, cov, beta_0=None, nu_0=None, m_0=None, W_0=None):
    if len(cluster_points.shape) == 2:
        dims = cluster_points.shape[1]
    elif len(cluster_points.shape) == 1:
        dims = 1
    else:
        raise Exception('Array much be of one or two dimensions!')
    if not beta_0:
        beta_0 = 1
    if not nu_0:
        nu_0 = dims
    if not m_0:
        m_0 = np.zeros(dims)
    if not W_0:
        W_0 = np.eye(dims)
    ll = np.sum(mvn_ll(cluster_points, mu, cov))
    ll += mvn_ll_single(mu, m_0, cov/beta_0)
    ll += inv_wishart_ll(cov, W_0, nu_0)
    return ll

## def pll(cluster_points, mu, cov):
##     # Get posterior parameters
##     beta_p, nu_p, m_p, W_p = post_norm_wish(cluster_points)
##     # Add wishart ll and norm ll
##     ll = wish.inv_wishart_ll(cov, W_p, nu_p)#pymc.inverse_wishart_like(cov, nu_p, np.linalg.inv(W_p))#
##     ll += mvn_ll_single(mu, m_p, cov/beta_p)
##     return ll

## def log_norm_const(cluster_points, mu, cov):
##     return jll(cluster_points, mu, cov)-pll(cluster_points, mu, cov)




if __name__ == "__main__":
    import pymc
    import pylab


    ################################################################
    # Set the parameters
    dims = 2.
    beta_0 = 1.
    nu_0 = dims
    m_0 = np.zeros(dims)
    W_0 = np.eye(dims)*1
    N_points = 50
    similarity = 1.


    # Generate some data for two distributions
    same = False
    prior_deg_freedom = nu_0 +similarity# must be >= dims
    prior_mu = m_0
    prior_cov = W_0/similarity
    prior_cov_wish = np.array(pymc.rwishart_cov(nu_0, W_0))#W_0#np.eye(dims)
    true_mu1 = pymc.rmv_normal_cov(prior_mu,prior_cov)
    true_cov1 = np.array(pymc.rwishart_cov(prior_deg_freedom, prior_cov_wish))
    true_mu2 = pymc.rmv_normal_cov(prior_mu,prior_cov)
    true_cov2 = np.array(pymc.rwishart_cov(prior_deg_freedom, prior_cov_wish))
    if same:
        true_mu2 = true_mu1
        true_cov2 = true_cov1
    #true_mu2 = true_mu1+0.1
    obs1 = pymc.rmv_normal_cov(true_mu1, true_cov1, size = N_points)
    obs2 = pymc.rmv_normal_cov(true_mu2, true_cov2, size = N_points)

    all_obs = np.vstack((obs1,obs2))
    all_labels = np.hstack((np.zeros(len(obs1)),np.ones(len(obs2))))

    pylab.figure()
    pylab.scatter(obs1[:,0],obs1[:,1])
    pylab.scatter(obs2[:,0],obs2[:,1], color='red')
    pylab.show()


    ###############################################################
    # Compare the posterior covariance to the maximum likelihood covariance
    # Sanity check

    beta_t, nu_t, m_t, W_t = post_norm_wish(all_obs[all_labels==0])

    C_t = W_t/(nu_t-W_t.shape[0]-1)

    print "Posterior expectation for the covariance"
    print C_t
    print "True covariance"
    print true_cov1
    print "Empirical covariance"
    print np.cov(all_obs[all_labels==0].T)

# Utilities

## def log_mean(array):
##     max_val = np.max(array)
##     return np.log(np.mean(np.exp(array-max_val))) + max_val


## def log_sum(array):
##     max_val = np.max(array)
##     return np.log(np.sum(np.exp(array-max_val))) + max_val


## def log_sum2(array1,array2):
##     combined = np.vstack((array1,array2))
##     max_val = np.max(combined,0)
##     return np.log(np.sum(np.exp(combined-max_val),0)) + max_val


